from __future__ import annotations

import os
from pathlib import Path
import re
import subprocess
import tempfile
import unittest

from gac.backends.vortex_buffered_mlir import (
    VortexBufferPlan,
    audit_bevformer_ffn_buffered_device_mlir,
    emit_bevformer_ffn_buffered_device_mlir,
)
from gac.backends.vortex_device_mlir import BEVFORMER_FFN_KERNELS
from gac.errors import BackendError


class VortexBufferedMlirTests(unittest.TestCase):
    contract = dict(rows=3, channels=19, hidden=7, epsilon=1.0e-5,
                    optimized_mlir_sha256="a" * 64,
                    call_plan_mlir_sha256="b" * 64)

    def audit(self, source: str, **overrides) -> None:
        contract = self.contract | overrides
        audit_bevformer_ffn_buffered_device_mlir(
            source, kernels=BEVFORMER_FFN_KERNELS, **contract)

    def test_plan_accounts_for_both_tiles_and_both_stages(self) -> None:
        plan = VortexBufferPlan(tile_m=3, tile_n=5, tile_k=7)
        self.assertEqual(plan.threads, 15)
        self.assertEqual(plan.lmem_bytes, 2 * (3 * 7 + 7 * 5) * 4)
        self.assertEqual(plan.grid_size(7, 11), 9)

    def test_plan_rejects_bad_descriptor_extents_and_incomplete_wgather(self) -> None:
        for args in ({"tile_m": 0}, {"tile_n": True}, {"tile_k": 65536},
                     {"tile_m": 1, "tile_n": 3},
                     {"tile_m": 65535, "tile_n": 65535, "tile_k": 65535}):
            with self.subTest(args=args), self.assertRaises(BackendError):
                VortexBufferPlan(**args)

    def test_audit_rejects_host_storage_and_collective_contract_drift(self) -> None:
        source = emit_bevformer_ffn_buffered_device_mlir(**self.contract)
        self.audit(source)
        mutations = (
            source.replace("@__gac_buffer_stages(2", "@__gac_buffer_stages(1"),
            source.replace("@__gac_buffer_lmem_bytes(512", "@__gac_buffer_lmem_bytes(256"),
            source.replace('"noinline", "optnone"', '"noinline"'),
            source.replace("={x31},i,0,r,r,r,i,~{memory}", "=r,i,0,r,r,r,i,~{memory}"),
            source.replace(".insn r $6, 0, 3, x0, $0, x0", ".insn r $6, 0, 2, x0, $0, x0"),
        )
        for index, changed in enumerate(mutations):
            with self.subTest(mutation=index), self.assertRaises(BackendError):
                self.audit(changed)

    def test_native_mlir_llvm_and_vortex_instruction_selection(self) -> None:
        tool_root = os.environ.get("GAC_VOLT_LLVM_PATH")
        if not tool_root:
            self.skipTest("set GAC_VOLT_LLVM_PATH to the pinned VOLT LLVM 20 build")
        tools = Path(tool_root) / "bin"
        with tempfile.TemporaryDirectory(prefix="gac-buffered-test-") as directory:
            root = Path(directory)
            source = root / "kernel.mlir"
            optimized = root / "kernel.opt.mlir"
            ir = root / "kernel.ll"
            optimized_ir = root / "kernel.opt.ll"
            assembly = root / "kernel.s"
            obj = root / "kernel.o"
            source.write_text(emit_bevformer_ffn_buffered_device_mlir(**self.contract), encoding="utf-8")

            def run(tool: str, *args: str) -> str:
                process = subprocess.run([str(tools / tool), *map(str, args)],
                                         capture_output=True, text=True, timeout=60)
                self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
                return process.stdout

            run("mlir-opt", "--verify-each", "--canonicalize", "--cse", source, "-o", optimized)
            self.audit(optimized.read_text(encoding="utf-8"))
            run("mlir-translate", "--mlir-to-llvmir", optimized, "-o", ir)
            run("opt", "-passes=verify", "-disable-output", ir)
            run("opt", "-passes=default<O2>", "-S", ir, "-o", optimized_ir)
            optimized_text = optimized_ir.read_text(encoding="utf-8")
            self.assertRegex(optimized_text, r"attributes #\d+ = \{[^}]*convergent[^}]*noinline[^}]*optnone")
            flags = ("--target=riscv32-unknown-elf", "-march=rv32imaf_xvortex_zicond",
                     "-mabi=ilp32f", "-O2", "-ffixed-x31")
            run("clang", *flags, "-S", ir, "-o", assembly)
            # These must be adjacent instructions: copying packed x31 between
            # them would update active lanes only, losing WGATHER lane data.
            asm = assembly.read_text(encoding="utf-8")
            self.assertRegex(asm, r"\.insn r4 43, 0, 0, t6,[^\n]+\n\s*\.insn r 11, 0, 3, zero, t6, zero")
            run("clang", *flags, "-c", ir, "-o", obj)
            symbols = run("llvm-nm", obj)
            for name in BEVFORMER_FFN_KERNELS:
                self.assertIn("__vx_kentry_" + name, symbols)


if __name__ == "__main__":
    unittest.main()
