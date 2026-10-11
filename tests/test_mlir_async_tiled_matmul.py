from __future__ import annotations

import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from gac.errors import BackendError
from gac.mlir_async_tiled_matmul import (
    ASYNC_MATMUL_BLOCK,
    ASYNC_MATMUL_K,
    ASYNC_MATMUL_M,
    ASYNC_MATMUL_N,
    ASYNC_MATMUL_SHARED_BYTES,
    ASYNC_MATMUL_SYMBOL,
    emit_async_tiled_matmul_gpu_mlir,
    inspect_async_tiled_matmul_ptx,
    lower_async_tiled_matmul_to_ptx,
)
from gac.mlir_toolchain import MlirPtxArtifact


def _artifact(source: str, ptx: str) -> MlirPtxArtifact:
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
        entries=(ASYNC_MATMUL_SYMBOL,),
    )


def _good_ptx() -> str:
    copies = (
        "  cp.async.ca.shared.global [%rd1], [%rd2], 16;\n"
        "  cp.async.ca.shared.global [%rd3], [%rd4], 16;\n"
    )
    consume = (
        "  bar.sync 0;\n"
        "  ld.shared.b32 %r1, [%rd1];\n"
        "  ld.shared.b32 %r2, [%rd2];\n"
        "  mul.rn.f32 %f1, %f2, %f3;\n"
        "  add.rn.f32 %f4, %f1, %f5;\n"
    )
    return f""".version 7.1
.target sm_86
.address_size 64
.visible .entry {ASYNC_MATMUL_SYMBOL}(
  .param .u64 .ptr .global a,
  .param .u64 .ptr .global b,
  .param .u64 .ptr .global c
)
.maxntid 64, 1, 1
{{
  .shared .align 16 .b8 shared_a[2048];
  .shared .align 16 .b8 shared_b[2048];
{copies}  cp.async.commit_group;
{copies}  cp.async.commit_group;
  cp.async.wait_group 1;
{consume}{copies}  cp.async.commit_group;
  cp.async.wait_group 1;
{consume}{copies}  cp.async.commit_group;
  cp.async.wait_group 1;
{consume}  cp.async.wait_group 0;
{consume}  bar.sync 0;
  bar.sync 0;
  bar.sync 0;
  st.global.b32 [%rd10], %r1;
  st.global.b32 [%rd10+4], %r2;
  st.global.b32 [%rd10+8], %r3;
  st.global.b32 [%rd10+12], %r4;
  ret;
}}
"""


class AsyncTiledMatmulEmitterTests(unittest.TestCase):
    def test_source_is_real_matmul_with_two_physical_ping_pong_stages(self):
        source = emit_async_tiled_matmul_gpu_mlir()
        self.assertIn(f"gpu.func @{ASYNC_MATMUL_SYMBOL}", source)
        self.assertIn("%a: memref<16x64xf32", source)
        self.assertIn("%b: memref<64x16xf32", source)
        self.assertIn("%c: memref<16x16xf32", source)
        self.assertEqual(
            source.count("memref<2x16x16xf32, #gpu.address_space<workgroup>>"),
            30,
        )
        self.assertEqual(source.count("nvgpu.device_async_copy"), 8)
        self.assertEqual(source.count("nvgpu.device_async_create_group"), 4)
        self.assertEqual(source.count("numGroups = 1"), 3)
        self.assertEqual(source.count("numGroups = 0"), 1)
        self.assertEqual(source.count("arith.mulf"), 16)
        self.assertEqual(source.count("arith.addf"), 16)
        self.assertEqual(source.count("memref.store %acc"), 4)
        self.assertNotIn("capability probe only", source)

    def test_stage_refills_follow_consumption_barriers(self):
        source = emit_async_tiled_matmul_gpu_mlir()
        first_wait = source.index("nvgpu.device_async_wait %group0 numGroups = 1")
        first_consume = source.index("%acc0_s0_first")
        refill_stage0 = source.index("%a_copy2")
        second_wait = source.index("nvgpu.device_async_wait %group1 numGroups = 1")
        second_consume = source.index("%acc0_s1_first")
        refill_stage1 = source.index("%a_copy3")
        final_wait = source.index("nvgpu.device_async_wait %group3 numGroups = 0")
        self.assertLess(first_wait, first_consume)
        self.assertLess(first_consume, refill_stage0)
        self.assertLess(refill_stage0, second_wait)
        self.assertLess(second_wait, second_consume)
        self.assertLess(second_consume, refill_stage1)
        self.assertLess(refill_stage1, final_wait)


class AsyncTiledMatmulLoweringTests(unittest.TestCase):
    def test_ptx_evidence_requires_full_pipeline_and_computation(self):
        evidence = inspect_async_tiled_matmul_ptx(_good_ptx())
        self.assertEqual(evidence.shared_declarations, 2)
        self.assertEqual(evidence.static_shared_bytes, ASYNC_MATMUL_SHARED_BYTES)
        self.assertEqual(evidence.async_copy_instructions, 8)
        self.assertEqual(evidence.async_commit_groups, 4)
        self.assertEqual(evidence.async_wait_group_depths, (1, 1, 1, 0))
        self.assertTrue(evidence.has_ping_pong_schedule)
        self.assertTrue(evidence.computes_real_matmul)

    def test_lowering_returns_fixed_executable_contract(self):
        source = emit_async_tiled_matmul_gpu_mlir()
        with patch(
            "gac.mlir_async_tiled_matmul.lower_gpu_mlir_to_ptx",
            return_value=_artifact(source, _good_ptx()),
        ) as lower:
            program = lower_async_tiled_matmul_to_ptx(
                command=("mock-mlir-opt",), timeout=43
            )
        self.assertEqual((program.m, program.n, program.k), (16, 16, 64))
        self.assertEqual(program.grid, (1, 1, 1))
        self.assertEqual(program.block, ASYNC_MATMUL_BLOCK)
        self.assertEqual(program.stages, 2)
        self.assertTrue(program.evidence.has_ping_pong_schedule)
        _, kwargs = lower.call_args
        self.assertEqual(tuple(kwargs["expected_entries"]), (ASYNC_MATMUL_SYMBOL,))
        self.assertEqual(tuple(kwargs["command"]), ("mock-mlir-opt",))
        self.assertEqual(kwargs["timeout"], 43)

    def test_lowering_fails_closed_on_pipeline_and_abi_regressions(self):
        source = emit_async_tiled_matmul_gpu_mlir()
        cases = (
            (_good_ptx().replace(".maxntid 64, 1, 1", ".maxntid 32, 1, 1"), "maxntid"),
            (_good_ptx().replace("cp.async.commit_group", "// missing", 1), "commit/wait order"),
            (_good_ptx().replace("cp.async.wait_group 0", "cp.async.wait_group 1"), "commit/wait order"),
            (_good_ptx().replace("mul.rn.f32", "mov.b32"), "arithmetic or global stores"),
            (_good_ptx().replace("st.global.b32", "st.shared.b32"), "arithmetic or global stores"),
        )
        for ptx, message in cases:
            with self.subTest(message=message), patch(
                "gac.mlir_async_tiled_matmul.lower_gpu_mlir_to_ptx",
                return_value=_artifact(source, ptx),
            ):
                with self.assertRaisesRegex(BackendError, message):
                    lower_async_tiled_matmul_to_ptx(command=("mock",))

    def test_shape_constants_are_exact(self):
        self.assertEqual((ASYNC_MATMUL_M, ASYNC_MATMUL_N, ASYNC_MATMUL_K), (16, 16, 64))


class AsyncTiledMatmulRunToolTests(unittest.TestCase):
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
                str(root / "tools" / "run_mlir_async_tiled_matmul.py"),
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
