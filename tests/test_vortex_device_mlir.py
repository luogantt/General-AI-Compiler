from __future__ import annotations

import unittest

from gac.backends.vortex_device_mlir import (
    BEVFORMER_FFN_KERNELS,
    CTA_BLOCK_DIM_X_CSR,
    CTA_BLOCK_ID_X_CSR,
    CTA_THREAD_ID_X_CSR,
    DATA_LAYOUT,
    TARGET_TRIPLE,
    VECTOR_ADD_KERNEL,
    audit_bevformer_ffn_device_mlir,
    emit_bevformer_ffn_device_mlir,
    emit_vector_add_device_mlir,
)
from gac.errors import BackendError


class VortexDeviceMlirTests(unittest.TestCase):
    def _emit_ffn(self) -> str:
        return emit_bevformer_ffn_device_mlir(
            rows=900,
            channels=256,
            hidden=512,
            epsilon=1.0e-5,
            optimized_mlir_sha256="A" * 64,
            call_plan_mlir_sha256="b" * 64,
        )

    def _audit_ffn(
        self,
        source: str,
        *,
        kernels: tuple[str, ...] = BEVFORMER_FFN_KERNELS,
    ) -> None:
        audit_bevformer_ffn_device_mlir(
            source,
            kernels=kernels,
            rows=900,
            channels=256,
            hidden=512,
            epsilon=1.0e-5,
            optimized_mlir_sha256="a" * 64,
            call_plan_mlir_sha256="b" * 64,
        )

    def test_vector_add_is_a_direct_rv32_vortex_module(self) -> None:
        source = emit_vector_add_device_mlir()

        self.assertIn(f'llvm.target_triple = "{TARGET_TRIPLE}"', source)
        self.assertIn(f'llvm.data_layout = "{DATA_LAYOUT}"', source)
        self.assertEqual(source.count(f"llvm.func @{VECTOR_ADD_KERNEL}"), 1)
        self.assertIn('passthrough = ["vortex-kernel"]', source)
        self.assertIn(f'section = ".vx_entry.{VECTOR_ADD_KERNEL}"', source)
        self.assertIn("@llvm.global.annotations", source)
        self.assertIn("@llvm.used", source)
        self.assertIn("@llvm.compiler.used", source)
        self.assertEqual(
            source.count(f"llvm.mlir.addressof @{VECTOR_ADD_KERNEL}"), 3
        )
        self.assertIn(
            "!llvm.struct<(i32, i32, i64, i64, i64)>", source
        )
        self.assertEqual(source.count('"csrr $0, $1", "=r,i"'), 3)
        for csr in (
            CTA_THREAD_ID_X_CSR,
            CTA_BLOCK_ID_X_CSR,
            CTA_BLOCK_DIM_X_CSR,
        ):
            self.assertIn(f"constant({csr} : i32)", source)
        self.assertEqual(source.count("llvm.inttoptr"), 3)
        self.assertEqual(source.count("llvm.trunc"), 3)
        self.assertIn("llvm.fadd", source)
        self.assertNotIn("#include", source)

    def test_ffn_emission_is_deterministic_bound_and_auditable(self) -> None:
        source = self._emit_ffn()

        self.assertEqual(source, self._emit_ffn())
        self._audit_ffn(source)
        self.assertIn("@__gac_plan_rows(900 : i32)", source)
        self.assertIn("@__gac_plan_channels(256 : i32)", source)
        self.assertIn("@__gac_plan_hidden(512 : i32)", source)
        self.assertIn("@__gac_plan_epsilon(1.000000000e-05 : f32)", source)
        self.assertIn('"' + "a" * 64 + "\\00", source)
        self.assertIn('"' + "b" * 64 + "\\00", source)
        for kernel in BEVFORMER_FFN_KERNELS:
            self.assertEqual(source.count(f"llvm.func @{kernel}"), 1)
            self.assertEqual(source.count(f"llvm.mlir.addressof @{kernel}"), 3)
        self.assertEqual(source.count('"csrr $0, $1", "=r,i"'), 9)
        self.assertIn('llvm.call_intrinsic "llvm.sqrt"', source)

    def test_ffn_uses_host_compatible_rv32_argument_blobs(self) -> None:
        source = self._emit_ffn()

        # Explicit i32 padding fixes the first i64 address at byte offset 16,
        # matching the C ABI structs used by the Vortex host launcher.
        self.assertIn(
            "!llvm.struct<(i32, i32, i32, i32, i64, i64, i64, i64)>",
            source,
        )
        self.assertIn(
            "!llvm.struct<(i32, i32, i32, i32, i64, i64, i64, i64, i64)>",
            source,
        )
        self.assertIn(
            "!llvm.struct<(i32, i32, f32, i32, i64, i64, i64, i64)>",
            source,
        )
        self.assertEqual(source.count("llvm.inttoptr"), 13)
        self.assertEqual(source.count("llvm.trunc"), 13)
        self.assertIn("%query_addr_ptr = llvm.getelementptr %args[0, 4]", source)
        self.assertIn("%activation_addr_ptr = llvm.getelementptr %args[0, 5]", source)
        self.assertIn("%input_addr_ptr = llvm.getelementptr %args[0, 4]", source)

    def test_emitter_rejects_values_that_break_the_rv32_contract(self) -> None:
        valid = dict(
            rows=900,
            channels=256,
            hidden=512,
            epsilon=1.0e-5,
            optimized_mlir_sha256="a" * 64,
            call_plan_mlir_sha256="b" * 64,
        )
        invalid_updates = (
            {"rows": 0},
            {"rows": True},
            {"channels": -1},
            {"hidden": 1 << 32},
            {"rows": 1 << 16, "channels": 1 << 16},
            {"rows": 1, "channels": 1 << 16, "hidden": 1 << 16},
            {"rows": 1, "channels": 1, "hidden": 1 << 30},
            {"epsilon": 0.0},
            {"epsilon": float("nan")},
            {"optimized_mlir_sha256": "xyz"},
            {"call_plan_mlir_sha256": "0" * 63},
        )
        for update in invalid_updates:
            with self.subTest(update=update):
                arguments = valid | update
                with self.assertRaises(BackendError):
                    emit_bevformer_ffn_device_mlir(**arguments)

        largest_addressable_fp32_buffer = valid | {
            "rows": 1,
            "channels": 1,
            "hidden": ((1 << 32) - 1) // 4,
        }
        self.assertIn(
            "@__gac_plan_hidden(1073741823 : i32)",
            emit_bevformer_ffn_device_mlir(**largest_addressable_fp32_buffer),
        )

    def test_audit_rejects_missing_or_substituted_abi_markers(self) -> None:
        source = self._emit_ffn()
        corruptions = (
            source.replace(TARGET_TRIPLE, "riscv64-unknown-elf", 1),
            source.replace("@llvm.global.annotations", "@bad.annotations", 1),
            source.replace("@llvm.used", "@bad.used", 1),
            source.replace("@llvm.compiler.used", "@bad.compiler.used", 1),
            source.replace("vortex.kernel\\00", "vortex.worker\\00", 1),
            source.replace(f"constant({CTA_THREAD_ID_X_CSR} : i32)", "constant(1 : i32)", 1),
            source.replace('passthrough = ["vortex-kernel"]', 'passthrough = ["ordinary"]', 1),
            source.replace(".vx_entry.bevformer_fc1_bias_relu", ".text", 1),
            source.replace('llvm.call_intrinsic "llvm.sqrt"', 'llvm.call_intrinsic "llvm.fabs"', 1),
            source.replace(
                "%record0_2 = llvm.insertvalue %annotation_text, %record0_1[1]",
                "%record0_2 = llvm.insertvalue %source_file, %record0_1[1]",
                1,
            ),
            source.replace(
                "llvm.return %used3 : !llvm.array<3 x ptr>",
                "llvm.return %used0 : !llvm.array<3 x ptr>",
                1,
            ),
            source.replace(
                "llvm.return %compiler_used3 : !llvm.array<3 x ptr>",
                "llvm.return %compiler_used0 : !llvm.array<3 x ptr>",
                1,
            ),
            source + "\n// kernel.cpp\n",
        )
        for corrupted in corruptions:
            with self.subTest(corruption=corrupted[-80:]):
                with self.assertRaises(BackendError):
                    self._audit_ffn(corrupted)

    def test_audit_rejects_changed_provenance_values(self) -> None:
        source = self._emit_ffn()
        corruptions = (
            source.replace("@__gac_plan_rows(900 : i32)", "@__gac_plan_rows(901 : i32)", 1),
            source.replace("@__gac_plan_channels(256 : i32)", "@__gac_plan_channels(128 : i32)", 1),
            source.replace("@__gac_plan_hidden(512 : i32)", "@__gac_plan_hidden(256 : i32)", 1),
            source.replace(
                "@__gac_plan_epsilon(1.000000000e-05 : f32)",
                "@__gac_plan_epsilon(2.000000000e-05 : f32)",
                1,
            ),
            source.replace("a" * 64 + "\\00", "c" * 64 + "\\00", 1),
            source.replace("b" * 64 + "\\00", "d" * 64 + "\\00", 1),
        )
        for corrupted in corruptions:
            with self.subTest(corruption=corrupted[-120:]):
                with self.assertRaises(BackendError):
                    self._audit_ffn(corrupted)

    def test_audit_rejects_changed_argument_fields_and_index_math(self) -> None:
        source = self._emit_ffn()
        corruptions = (
            source.replace(
                "%query_addr_ptr = llvm.getelementptr %args[0, 4]",
                "%query_addr_ptr = llvm.getelementptr %args[0, 5]",
                1,
            ),
            source.replace(
                "%weight_index = llvm.add %weight_row, %column : i32",
                "%weight_index = llvm.add %weight_row, %row : i32",
                1,
            ),
            source.replace(
                "%activation_index = llvm.add %activation_row, %k : i32",
                "%activation_index = llvm.add %activation_row, %column : i32",
                1,
            ),
            source.replace(
                "%variance_epsilon = llvm.fadd %variance, %epsilon : f32",
                "%variance_epsilon = llvm.fadd %mean, %epsilon : f32",
                1,
            ),
        )
        for corrupted in corruptions:
            with self.subTest(corruption=corrupted[-120:]):
                with self.assertRaises(BackendError):
                    self._audit_ffn(corrupted)

    def test_audit_rejects_a_different_runtime_plan(self) -> None:
        source = self._emit_ffn()
        with self.assertRaises(BackendError):
            self._audit_ffn(source, kernels=BEVFORMER_FFN_KERNELS[:-1])
        with self.assertRaises(BackendError):
            self._audit_ffn(source, kernels=(BEVFORMER_FFN_KERNELS[0],) * 3)


if __name__ == "__main__":
    unittest.main()
