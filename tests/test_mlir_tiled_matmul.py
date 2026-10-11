from __future__ import annotations

from dataclasses import replace
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from gac.errors import BackendError
from gac.mlir_tiled_matmul import (
    ASYNC_DOUBLE_BUFFER_SHARED_BYTES,
    ASYNC_DOUBLE_BUFFER_SYMBOL,
    FUNCTIONAL_MMA_FP16_K_SLICES,
    FUNCTIONAL_MMA_FP16_SYMBOL,
    FUNCTIONAL_MMA_TF32_SYMBOL,
    FUNCTIONAL_MMA_K_SLICES,
    TiledMatmulConfig,
    bevformer_fc1_tiled_config,
    bevformer_fc2_tiled_config,
    bevformer_tiled_candidates,
    emit_nvgpu_async_double_buffer_smoke_gpu_mlir,
    emit_nvgpu_mma_smoke_gpu_mlir,
    emit_nvgpu_fp16_matmul_gpu_mlir,
    emit_nvgpu_tf32_matmul_gpu_mlir,
    emit_tiled_matmul_gpu_mlir,
    inspect_tiled_matmul_ptx,
    inspect_functional_mma_ptx,
    lower_tiled_matmul_to_ptx,
    probe_nvgpu_async_double_buffer_to_ptx,
    probe_nvgpu_mma_to_ptx,
    probe_nvgpu_fp16_matmul_to_ptx,
    probe_nvgpu_tf32_matmul_to_ptx,
)
from gac.mlir_toolchain import MlirPtxArtifact


def _artifact(source: str, ptx: str, symbol: str) -> MlirPtxArtifact:
    return MlirPtxArtifact(
        binary_mlir="module { gpu.binary @m [] }",
        ptx=ptx,
        source_sha256=hashlib.sha256(source.encode("utf-8")).hexdigest(),
        command=("mlir-opt",),
        mlir_opt_version="LLVM 24.0.0git",
        mlir_opt_source_revision="0" * 40,
        target_chip="sm_86",
        target_features="+ptx71",
        ptx_version="7.1",
        entries=(symbol,),
    )


def _tiled_ptx(
    symbol: str, *, parameter_count: int = 3, maxntid: str = "16, 16, 1"
) -> str:
    parameters = ",\n".join(
        f"  .param .u64 .ptr .global {symbol}_param_{index}"
        for index in range(parameter_count)
    )
    return f""".version 7.1
.target sm_86
.address_size 64
.visible .entry {symbol}(
{parameters}
)
.maxntid {maxntid}
{{
  .shared .align 4 .b8 a[1024];
  .shared .align 4 .b8 b[1024];
  ld.shared.v4.b32 {{%r1,%r2,%r3,%r4}}, [%rd1];
  ld.shared.b32 %r5, [%rd2];
  st.shared.b32 [%rd1], %r1;
  bar.sync 0;
  mul.rn.f32 %f1, %f2, %f3;
  add.rn.f32 %f4, %f1, %f5;
  st.global.b32 [%rd3], %r1;
  bar.sync 0;
  ret;
}}
"""


def _async_double_buffer_ptx() -> str:
    return f""".version 7.1
.target sm_86
.address_size 64
.visible .entry {ASYNC_DOUBLE_BUFFER_SYMBOL}(
  .param .u64 .ptr .global input,
  .param .u64 .ptr .global output
)
.maxntid 4, 1, 1
{{
  .shared .align 4 .b8 stages[128];
  cp.async.ca.shared.global [%rd1], [%rd2], 16;
  cp.async.commit_group;
  cp.async.ca.shared.global [%rd1+64], [%rd2+64], 16;
  cp.async.commit_group;
  cp.async.wait_group 1;
  bar.sync 0;
  ld.shared.b32 %r1, [%rd1];
  st.global.b32 [%rd3], %r1;
  cp.async.wait_group 0;
  bar.sync 0;
  ld.shared.b32 %r2, [%rd1+64];
  st.global.b32 [%rd3+64], %r2;
  ret;
}}
"""


def _functional_tf32_mma_ptx(
    *, parameter_count: int = 3, maxntid: str = "32, 1, 1"
) -> str:
    parameters = ",\n".join(
        f"  .param .u64 .ptr .global p{index}" for index in range(parameter_count)
    )
    loads = "".join(
        f"  ld.global.b32 %r{index + 1}, [%rd{index + 1}];\n"
        for index in range(3 * FUNCTIONAL_MMA_K_SLICES)
    )
    mmas = "".join(
        "  mma.sync.aligned.m16n8k4.row.col.f32.tf32.tf32.f32;\n"
        for _ in range(FUNCTIONAL_MMA_K_SLICES)
    )
    return f""".version 7.1
.target sm_86
.address_size 64
.visible .entry {FUNCTIONAL_MMA_TF32_SYMBOL}(
{parameters}
)
.maxntid {maxntid}
{{
{loads}{mmas}  st.global.b32 [%rd4], %r4;
  st.global.b32 [%rd4+4], %r5;
  st.global.b32 [%rd4+8], %r6;
  st.global.b32 [%rd4+12], %r7;
  ret;
}}
"""


def _functional_fp16_mma_ptx(
    *, parameter_count: int = 3, maxntid: str = "32, 1, 1"
) -> str:
    parameters = ",\n".join(
        f"  .param .u64 .ptr .global p{index}" for index in range(parameter_count)
    )
    loads = "".join(
        f"  ld.global.u16 %r{index + 1}, [%rd{index + 1}];\n"
        for index in range(12 * FUNCTIONAL_MMA_FP16_K_SLICES)
    )
    mmas = "".join(
        "  mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32;\n"
        for _ in range(FUNCTIONAL_MMA_FP16_K_SLICES)
    )
    return f""".version 7.1
.target sm_86
.address_size 64
.visible .entry {FUNCTIONAL_MMA_FP16_SYMBOL}(
{parameters}
)
.maxntid {maxntid}
{{
{loads}{mmas}  st.global.b32 [%rd4], %r4;
  st.global.b32 [%rd4+4], %r5;
  st.global.b32 [%rd4+8], %r6;
  st.global.b32 [%rd4+12], %r7;
  ret;
}}
"""


class TiledMatmulConfigTests(unittest.TestCase):
    def test_production_configs_have_exact_shape_and_launch(self):
        fc1 = bevformer_fc1_tiled_config()
        self.assertEqual((fc1.m, fc1.k, fc1.n), (2500, 256, 512))
        self.assertTrue(fc1.fuse_bias_relu)
        self.assertEqual(fc1.grid, (32, 157, 1))
        self.assertEqual(fc1.block, (16, 16, 1))
        self.assertEqual(fc1.shared_memory_bytes, 2048)

        fc2 = bevformer_fc2_tiled_config()
        self.assertEqual((fc2.m, fc2.k, fc2.n), (2500, 512, 256))
        self.assertFalse(fc2.fuse_bias_relu)
        self.assertEqual(fc2.grid, (16, 157, 1))

    def test_candidate_tiles_are_all_valid_for_both_matmuls(self):
        for tile_m, tile_n, tile_k in bevformer_tiled_candidates():
            for factory in (bevformer_fc1_tiled_config, bevformer_fc2_tiled_config):
                config = factory(tile_m=tile_m, tile_n=tile_n, tile_k=tile_k)
                self.assertLessEqual(config.threads_per_block, 1024)
                self.assertLessEqual(config.shared_memory_bytes, 48 * 1024)

    def test_invalid_contracts_fail_before_emission(self):
        with self.assertRaisesRegex(ValueError, "identifier"):
            TiledMatmulConfig("bad-name", 1, 1, 1, 1, 1, 1, 1, 1)
        with self.assertRaisesRegex(ValueError, "positive integer"):
            replace(bevformer_fc1_tiled_config(), tile_k=0)
        with self.assertRaisesRegex(ValueError, "block_x == tile_n"):
            replace(bevformer_fc1_tiled_config(), block_x=8)
        with self.assertRaisesRegex(ValueError, "1024-thread"):
            TiledMatmulConfig("too_many", 1, 1, 1, 33, 32, 1, 32, 33)
        with self.assertRaisesRegex(ValueError, "48 KiB"):
            TiledMatmulConfig("too_shared", 1, 1, 1, 32, 32, 256, 32, 32)


class TiledMatmulEmitterTests(unittest.TestCase):
    def test_fc1_emits_real_workgroup_attributions_and_fused_epilogue(self):
        source = emit_tiled_matmul_gpu_mlir(bevformer_fc1_tiled_config())
        self.assertIn("workgroup(\n", source)
        self.assertEqual(source.count("#gpu.address_space<workgroup>"), 6)
        self.assertNotIn("memref.alloc", source)
        self.assertNotIn("memref.alloca", source)
        self.assertIn("scf.for %a_linear", source)
        self.assertIn("scf.for %b_linear", source)
        self.assertEqual(source.count("gpu.barrier"), 2)
        self.assertIn("arith.maximumf", source)
        self.assertIn("memref.load %bias", source)
        self.assertIn("arith.mulf", source)
        self.assertIn("arith.addf %inner_acc", source)
        self.assertNotIn("math.fma", source)
        self.assertNotIn("cublas", source.lower())

    def test_fc2_omits_bias_pointer_and_relu(self):
        source = emit_tiled_matmul_gpu_mlir(bevformer_fc2_tiled_config())
        signature = source.split("kernel attributes", 1)[0]
        self.assertNotIn("%bias", signature)
        self.assertNotIn("arith.maximumf", source)
        self.assertIn("%result = arith.addf %outer_acc, %zero", source)

    def test_partial_tiles_are_guarded_without_divergent_barriers(self):
        source = emit_tiled_matmul_gpu_mlir(bevformer_fc1_tiled_config())
        store_guard = source.index("scf.if %output_ok")
        self.assertGreater(store_guard, source.rindex("gpu.barrier"))
        self.assertIn("%a_ok = arith.andi", source)
        self.assertIn("%b_ok = arith.andi", source)


class TiledMatmulLoweringTests(unittest.TestCase):
    def test_instruction_evidence_distinguishes_vectorization_and_mma(self):
        ptx = _tiled_ptx("kernel") + "\nmma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32;\n"
        evidence = inspect_tiled_matmul_ptx(ptx)
        self.assertEqual(evidence.shared_declarations, 2)
        self.assertEqual(evidence.static_shared_bytes, 2048)
        self.assertEqual(evidence.shared_loads, 2)
        self.assertEqual(evidence.vector_shared_loads, 1)
        self.assertEqual(evidence.shared_stores, 1)
        self.assertEqual(evidence.global_stores, 1)
        self.assertEqual(evidence.barriers, 2)
        self.assertEqual(evidence.fp32_multiplies, 1)
        self.assertEqual(evidence.fp32_adds, 1)
        self.assertTrue(evidence.computes_real_matmul)
        self.assertTrue(evidence.uses_tensor_cores)
        self.assertFalse(evidence.uses_async_global_to_shared_copy)

    def test_lowering_passes_exact_entry_and_preserves_launch(self):
        config = bevformer_fc1_tiled_config()

        def fake(source: str, **_kwargs):
            return _artifact(
                source,
                _tiled_ptx(config.symbol, parameter_count=4),
                config.symbol,
            )

        with patch("gac.mlir_tiled_matmul.lower_gpu_mlir_to_ptx", side_effect=fake) as lower:
            program = lower_tiled_matmul_to_ptx(
                config, command=("mock-mlir-opt",), timeout=19
            )
        self.assertEqual(program.grid, config.grid)
        self.assertEqual(program.block, config.block)
        self.assertFalse(program.evidence.uses_tensor_cores)
        _, kwargs = lower.call_args
        self.assertEqual(tuple(kwargs["expected_entries"]), (config.symbol,))
        self.assertEqual(tuple(kwargs["command"]), ("mock-mlir-opt",))
        self.assertEqual(kwargs["timeout"], 19)

    def test_lowering_rejects_pointer_count_and_launch_abi_mismatches(self):
        config = bevformer_fc1_tiled_config()
        source = emit_tiled_matmul_gpu_mlir(config)
        with patch(
            "gac.mlir_tiled_matmul.lower_gpu_mlir_to_ptx",
            return_value=_artifact(
                source, _tiled_ptx(config.symbol, parameter_count=3), config.symbol
            ),
        ):
            with self.assertRaisesRegex(BackendError, "exactly 4 bare global pointers"):
                lower_tiled_matmul_to_ptx(config, command=("mock",))

        with patch(
            "gac.mlir_tiled_matmul.lower_gpu_mlir_to_ptx",
            return_value=_artifact(
                source,
                _tiled_ptx(
                    config.symbol, parameter_count=4, maxntid="128, 1, 1"
                ),
                config.symbol,
            ),
        ):
            with self.assertRaisesRegex(BackendError, "maxntid"):
                lower_tiled_matmul_to_ptx(config, command=("mock",))

    def test_lowering_rejects_fake_or_missing_shared_memory(self):
        config = bevformer_fc2_tiled_config()
        bad_ptx = _tiled_ptx(config.symbol).replace("  .shared", "  .local", 1)
        with patch(
            "gac.mlir_tiled_matmul.lower_gpu_mlir_to_ptx",
            return_value=_artifact("unused", bad_ptx, config.symbol),
        ):
            with self.assertRaisesRegex(BackendError, "exactly two"):
                lower_tiled_matmul_to_ptx(config, command=("mock",))

        local_ptx = _tiled_ptx(config.symbol) + "\n.local .b8 local_depot[16];\n"
        with patch(
            "gac.mlir_tiled_matmul.lower_gpu_mlir_to_ptx",
            return_value=_artifact("unused", local_ptx, config.symbol),
        ):
            with self.assertRaisesRegex(BackendError, "local allocation"):
                lower_tiled_matmul_to_ptx(config, command=("mock",))

    def test_lowering_rejects_missing_matmul_arithmetic_or_output_store(self):
        config = bevformer_fc2_tiled_config()
        source = emit_tiled_matmul_gpu_mlir(config)
        cases = (
            (
                _tiled_ptx(config.symbol)
                .replace("  mul.rn.f32 %f1, %f2, %f3;\n", "")
                .replace("  add.rn.f32 %f4, %f1, %f5;\n", ""),
                "floating-point accumulation",
            ),
            (
                _tiled_ptx(config.symbol).replace(
                    "  st.global.b32 [%rd3], %r1;\n", ""
                ),
                "global output store",
            ),
        )
        for ptx, message in cases:
            with self.subTest(message=message), patch(
                "gac.mlir_tiled_matmul.lower_gpu_mlir_to_ptx",
                return_value=_artifact(source, ptx, config.symbol),
            ):
                with self.assertRaisesRegex(BackendError, message):
                    lower_tiled_matmul_to_ptx(config, command=("mock",))


class NvgpuAsyncDoubleBufferProbeTests(unittest.TestCase):
    def test_source_has_two_physical_stages_and_overlap_wait_schedule(self):
        source = emit_nvgpu_async_double_buffer_smoke_gpu_mlir()
        self.assertIn(
            "%stages: memref<2x16xf32, #gpu.address_space<workgroup>>", source
        )
        self.assertEqual(source.count("nvgpu.device_async_copy"), 2)
        self.assertEqual(source.count("nvgpu.device_async_create_group"), 2)
        wait_one = source.index("nvgpu.device_async_wait %group0 numGroups = 1")
        stage_zero_load = source.index("%stage0 = memref.load")
        wait_zero = source.index("nvgpu.device_async_wait %group1 numGroups = 0")
        self.assertLess(wait_one, stage_zero_load)
        self.assertLess(stage_zero_load, wait_zero)
        self.assertIn("capability probe only", source)

    def test_instruction_evidence_records_async_groups_and_wait_depths(self):
        evidence = inspect_tiled_matmul_ptx(_async_double_buffer_ptx())
        self.assertEqual(evidence.shared_declarations, 1)
        self.assertEqual(evidence.static_shared_bytes, ASYNC_DOUBLE_BUFFER_SHARED_BYTES)
        self.assertEqual(evidence.async_copy_instructions, 2)
        self.assertEqual(evidence.async_commit_groups, 2)
        self.assertEqual(evidence.async_wait_groups, 2)
        self.assertEqual(evidence.async_wait_group_depths, (1, 0))
        self.assertTrue(evidence.uses_async_global_to_shared_copy)
        self.assertTrue(evidence.has_two_stage_async_pipeline)

    def test_probe_requires_double_stage_ptx_and_overlap_window(self):
        source = emit_nvgpu_async_double_buffer_smoke_gpu_mlir()
        good_ptx = _async_double_buffer_ptx()
        with patch(
            "gac.mlir_tiled_matmul.lower_gpu_mlir_to_ptx",
            return_value=_artifact(source, good_ptx, ASYNC_DOUBLE_BUFFER_SYMBOL),
        ) as lower:
            probe = probe_nvgpu_async_double_buffer_to_ptx(command=("mock",))
        self.assertEqual(probe.stage_count, 2)
        self.assertEqual(probe.stage_elements, 16)
        self.assertEqual(probe.shared_memory_bytes, 128)
        self.assertTrue(probe.evidence.has_two_stage_async_pipeline)
        _, kwargs = lower.call_args
        self.assertEqual(
            tuple(kwargs["expected_entries"]), (ASYNC_DOUBLE_BUFFER_SYMBOL,)
        )

        wrong_launch = good_ptx.replace(".maxntid 4, 1, 1", ".maxntid 32, 1, 1")
        with patch(
            "gac.mlir_tiled_matmul.lower_gpu_mlir_to_ptx",
            return_value=_artifact(source, wrong_launch, ASYNC_DOUBLE_BUFFER_SYMBOL),
        ):
            with self.assertRaisesRegex(BackendError, "maxntid 4, 1, 1"):
                probe_nvgpu_async_double_buffer_to_ptx(command=("mock",))

        no_overlap = good_ptx.replace(
            "  ld.shared.b32 %r1, [%rd1];\n  st.global.b32 [%rd3], %r1;\n"
            "  cp.async.wait_group 0;",
            "  cp.async.wait_group 0;\n"
            "  ld.shared.b32 %r1, [%rd1];\n  st.global.b32 [%rd3], %r1;",
        )
        with patch(
            "gac.mlir_tiled_matmul.lower_gpu_mlir_to_ptx",
            return_value=_artifact(source, no_overlap, ASYNC_DOUBLE_BUFFER_SYMBOL),
        ):
            with self.assertRaisesRegex(BackendError, "consume stage 0"):
                probe_nvgpu_async_double_buffer_to_ptx(command=("mock",))

        one_stage = good_ptx.replace("stages[128]", "stages[64]")
        with patch(
            "gac.mlir_tiled_matmul.lower_gpu_mlir_to_ptx",
            return_value=_artifact(source, one_stage, ASYNC_DOUBLE_BUFFER_SYMBOL),
        ):
            with self.assertRaisesRegex(BackendError, "exactly 128 bytes"):
                probe_nvgpu_async_double_buffer_to_ptx(command=("mock",))


class NvgpuMmaProbeTests(unittest.TestCase):
    def test_smoke_sources_encode_fp16_and_tf32_contracts(self):
        fp16 = emit_nvgpu_mma_smoke_gpu_mlir("fp16")
        self.assertIn("mmaShape = [16, 8, 16]", fp16)
        self.assertIn("vector<4x2xf16>", fp16)
        self.assertNotIn("tf32Enabled", fp16)

        tf32 = emit_nvgpu_mma_smoke_gpu_mlir("tf32")
        self.assertIn("mmaShape = [16, 8, 4] tf32Enabled", tf32)
        self.assertIn("vector<2x1xf32>", tf32)
        with self.assertRaisesRegex(ValueError, "fp16.*tf32"):
            emit_nvgpu_mma_smoke_gpu_mlir("fp8")

    def test_probe_requires_actual_ptx_mma_instruction(self):
        mode = "fp16"
        source = emit_nvgpu_mma_smoke_gpu_mlir(mode)
        symbol = "gac_nvgpu_mma_fp16_smoke"
        good_ptx = (
            ".version 7.1\n.target sm_86\n.address_size 64\n"
            ".visible .entry gac_nvgpu_mma_fp16_smoke() {\n"
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32;\nret;\n}\n"
        )
        with patch(
            "gac.mlir_tiled_matmul.lower_gpu_mlir_to_ptx",
            return_value=_artifact(source, good_ptx, symbol),
        ):
            probe = probe_nvgpu_mma_to_ptx(mode, command=("mock",))
        self.assertTrue(probe.evidence.uses_tensor_cores)

        no_mma = good_ptx.replace("mma.sync", "mul.rn.f32")
        with patch(
            "gac.mlir_tiled_matmul.lower_gpu_mlir_to_ptx",
            return_value=_artifact(source, no_mma, symbol),
        ):
            with self.assertRaisesRegex(BackendError, "no PTX mma.sync"):
                probe_nvgpu_mma_to_ptx(mode, command=("mock",))


class NvgpuFunctionalMmaProbeTests(unittest.TestCase):
    def test_source_loads_real_inputs_accumulates_k_and_stores_full_tile(self):
        source = emit_nvgpu_tf32_matmul_gpu_mlir()
        self.assertIn(f"gpu.func @{FUNCTIONAL_MMA_TF32_SYMBOL}", source)
        self.assertIn("scf.for %k_base", source)
        self.assertIn("iter_args(%acc = %zero_acc)", source)
        self.assertEqual(source.count("memref.load %a"), 2)
        self.assertEqual(source.count("memref.load %b"), 1)
        self.assertEqual(source.count("memref.store %c"), 4)
        self.assertIn("mmaShape = [16, 8, 4] tf32Enabled", source)
        self.assertIn("known_block_size = array<i32: 32, 1, 1>", source)
        self.assertNotIn("dense<1.0>", source)

    def test_fp16_source_loads_real_inputs_accumulates_two_k_slices(self):
        source = emit_nvgpu_fp16_matmul_gpu_mlir()
        self.assertIn(f"gpu.func @{FUNCTIONAL_MMA_FP16_SYMBOL}", source)
        self.assertIn("scf.for %k_base", source)
        self.assertIn("iter_args(%acc = %zero_acc)", source)
        self.assertEqual(source.count("memref.load %a"), 8)
        self.assertEqual(source.count("memref.load %b"), 4)
        self.assertEqual(source.count("memref.store %c"), 4)
        self.assertIn("mmaShape = [16, 8, 16]", source)
        self.assertIn("vector<4x2xf16>", source)
        self.assertIn("vector<2x2xf32>", source)
        self.assertIn("memref<16x32xf16", source)
        self.assertNotIn("dense<1.0>", source)

    def test_evidence_requires_global_io_around_exact_tf32_mma(self):
        evidence = inspect_functional_mma_ptx(_functional_tf32_mma_ptx())
        self.assertEqual(evidence.global_loads, 12)
        self.assertEqual(evidence.global_stores, 4)
        self.assertEqual(evidence.mma_instructions, 4)
        self.assertEqual(evidence.tf32_mma_instructions, 4)
        self.assertEqual(evidence.fp16_mma_instructions, 0)
        self.assertTrue(evidence.uses_real_global_inputs)
        self.assertTrue(evidence.writes_global_output)

        constants_only = "\n".join(
            line
            for line in _functional_tf32_mma_ptx().splitlines()
            if "ld.global" not in line
        )
        self.assertFalse(
            inspect_functional_mma_ptx(constants_only).uses_real_global_inputs
        )

    def test_fp16_evidence_and_probe_require_two_real_k_slices(self):
        source = emit_nvgpu_fp16_matmul_gpu_mlir()
        good_ptx = _functional_fp16_mma_ptx()
        evidence = inspect_functional_mma_ptx(good_ptx)
        self.assertEqual(evidence.global_loads, 24)
        self.assertEqual(evidence.global_stores, 4)
        self.assertEqual(evidence.mma_instructions, 2)
        self.assertEqual(evidence.fp16_mma_instructions, 2)
        self.assertEqual(evidence.tf32_mma_instructions, 0)

        with patch(
            "gac.mlir_tiled_matmul.lower_gpu_mlir_to_ptx",
            return_value=_artifact(source, good_ptx, FUNCTIONAL_MMA_FP16_SYMBOL),
        ) as lower:
            probe = probe_nvgpu_fp16_matmul_to_ptx(
                command=("mock-mlir-opt",), timeout=43
            )
        self.assertEqual(probe.mode, "fp16")
        self.assertEqual((probe.m, probe.n, probe.k), (16, 8, 32))
        self.assertEqual(probe.k_step, 16)
        self.assertEqual(probe.input_dtype, "float16")
        self.assertEqual(probe.output_dtype, "float32")
        self.assertEqual(probe.grid, (1, 1, 1))
        self.assertEqual(probe.block, (32, 1, 1))
        _, kwargs = lower.call_args
        self.assertEqual(
            tuple(kwargs["expected_entries"]), (FUNCTIONAL_MMA_FP16_SYMBOL,)
        )
        self.assertEqual(tuple(kwargs["command"]), ("mock-mlir-opt",))
        self.assertEqual(kwargs["timeout"], 43)

    def test_fp16_probe_rejects_constant_fragment_or_wrong_mma_signature(self):
        source = emit_nvgpu_fp16_matmul_gpu_mlir()
        no_real_inputs = "\n".join(
            line
            for line in _functional_fp16_mma_ptx().splitlines()
            if "ld.global" not in line
        )
        wrong_mma = _functional_fp16_mma_ptx().replace(
            ".f16.f16.f32", ".tf32.tf32.f32"
        )
        for ptx, message in (
            (no_real_inputs, "load real A/B values"),
            (wrong_mma, "f32.f16.f16.f32"),
        ):
            with self.subTest(message=message), patch(
                "gac.mlir_tiled_matmul.lower_gpu_mlir_to_ptx",
                return_value=_artifact(source, ptx, FUNCTIONAL_MMA_FP16_SYMBOL),
            ):
                with self.assertRaisesRegex(BackendError, message):
                    probe_nvgpu_fp16_matmul_to_ptx(command=("mock",))

    def test_probe_records_executable_contract_and_forwards_tool_options(self):
        source = emit_nvgpu_tf32_matmul_gpu_mlir()
        with patch(
            "gac.mlir_tiled_matmul.lower_gpu_mlir_to_ptx",
            return_value=_artifact(
                source, _functional_tf32_mma_ptx(), FUNCTIONAL_MMA_TF32_SYMBOL
            ),
        ) as lower:
            probe = probe_nvgpu_tf32_matmul_to_ptx(
                command=("mock-mlir-opt",), timeout=41
            )
        self.assertEqual((probe.m, probe.n, probe.k), (16, 8, 16))
        self.assertEqual(probe.k_step, 4)
        self.assertEqual(probe.grid, (1, 1, 1))
        self.assertEqual(probe.block, (32, 1, 1))
        self.assertEqual(probe.input_dtype, "float32")
        self.assertTrue(probe.evidence.uses_real_global_inputs)
        _, kwargs = lower.call_args
        self.assertEqual(
            tuple(kwargs["expected_entries"]), (FUNCTIONAL_MMA_TF32_SYMBOL,)
        )
        self.assertEqual(tuple(kwargs["command"]), ("mock-mlir-opt",))
        self.assertEqual(kwargs["timeout"], 41)

    def test_probe_fails_closed_on_abi_and_codegen_regressions(self):
        source = emit_nvgpu_tf32_matmul_gpu_mlir()
        cases = (
            (
                _functional_tf32_mma_ptx(parameter_count=2),
                "exactly three bare global pointers",
            ),
            (_functional_tf32_mma_ptx(maxntid="16, 1, 1"), "maxntid 32"),
            (
                _functional_tf32_mma_ptx().replace("ld.global", "ld.shared"),
                "load real A/B values",
            ),
            (
                _functional_tf32_mma_ptx().replace("st.global", "st.shared"),
                "store all four",
            ),
            (
                _functional_tf32_mma_ptx().replace(".tf32.tf32", ".f16.f16"),
                "f32.tf32.tf32.f32",
            ),
        )
        for ptx, message in cases:
            with self.subTest(message=message), patch(
                "gac.mlir_tiled_matmul.lower_gpu_mlir_to_ptx",
                return_value=_artifact(source, ptx, FUNCTIONAL_MMA_TF32_SYMBOL),
            ):
                with self.assertRaisesRegex(BackendError, message):
                    probe_nvgpu_tf32_matmul_to_ptx(command=("mock",))


class FunctionalMmaRunToolTests(unittest.TestCase):
    @staticmethod
    def _run(*arguments: str) -> subprocess.CompletedProcess[str]:
        root = Path(__file__).resolve().parents[1]
        environment = dict(os.environ)
        source = str(root / "src")
        environment["PYTHONPATH"] = (
            source
            if not environment.get("PYTHONPATH")
            else source + os.pathsep + environment["PYTHONPATH"]
        )
        return subprocess.run(
            [
                sys.executable,
                str(root / "tools" / "run_mlir_functional_mma.py"),
                *arguments,
            ],
            cwd=root,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )

    def test_report_cannot_overwrite_or_hardlink_alias_input_ptx(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ptx = root / "input.ptx"
            original = "input evidence must survive\n"
            ptx.write_text(original, encoding="utf-8")

            exact = self._run(
                "--ptx", str(ptx), "--output-json", str(ptx)
            )
            self.assertNotEqual(exact.returncode, 0)
            self.assertIn("must not overwrite or alias", exact.stderr)
            self.assertEqual(ptx.read_text(encoding="utf-8"), original)

            alias = root / "report.json"
            try:
                alias.hardlink_to(ptx)
            except OSError as exc:
                self.skipTest(f"hard links are unavailable: {exc}")
            linked = self._run(
                "--ptx", str(ptx), "--output-json", str(alias)
            )
            self.assertNotEqual(linked.returncode, 0)
            self.assertIn("must not overwrite or alias", linked.stderr)
            self.assertEqual(ptx.read_text(encoding="utf-8"), original)


if __name__ == "__main__":
    unittest.main()
