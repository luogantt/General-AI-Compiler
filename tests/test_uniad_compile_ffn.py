"""CLI coverage for UniAD FFN deployment artifact selection."""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from gac.tools import uniad_compile_ffn as compiler
from gac.uniad import UniADFFNContract, write_bundle


class UniADCompileFfnTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.bundle = self.root / "layer2.uniadffn"
        contract = UniADFFNContract(layer=2)
        weights = {
            name: np.zeros(shape, dtype=np.float32)
            for name, shape in contract.weight_shapes.items()
        }
        write_bundle(self.bundle, contract, weights)

    @staticmethod
    def _write_fake_package(path: Path, *_args, **_kwargs) -> None:
        path.write_bytes(b"verified-hxp")

    @staticmethod
    def _write_offline_manifest(
        directory: Path,
        *,
        tiled: bool,
    ) -> dict:
        directory.mkdir(parents=True)
        artifact_files = (
            compiler.MLIR_TILED_FFN_FILES
            if tiled
            else compiler.MLIR_CUBLAS_PTX_FILES
        )
        records = {}
        for key, name in artifact_files.items():
            payload = (key + "\n").encode("utf-8")
            (directory / name).write_bytes(payload)
            records[key] = {
                "file": name,
                "size_bytes": len(payload),
                "sha256": compiler._sha256_bytes(payload),
            }
        manifest = {
            "artifact": (
                "gac.mlir-tiled-ffn-sm86.directory"
                if tiled
                else "gac.mlir-cublas-ptx-sm86.directory"
            ),
            "artifact_schema_version": 1,
            "compiler_backend": (
                compiler.MLIR_TILED_FFN_TARGET
                if tiled
                else compiler.MLIR_CUBLAS_PTX_TARGET
            ),
            "deployment_boundary": "weight-free validated PTX",
            "executable": True,
            "executed_custom_artifact": "textual-ptx",
            "ptx_dispatch_count": 3 if tiled else 2,
            "nvrtc_required": False,
            "matmul_lowering": (
                "explicit-shared-memory-tiled-mlir-gpu"
                if tiled
                else "cupy-cublas"
            ),
            "fusion_lowering": "mlir-gpu-nvvm-ptx",
            "dispatch_count": 3 if tiled else 4,
            "integrity": {"integrity_sha256": "a" * 64},
            "manifest_payload_sha256": "b" * 64,
            "files": records,
            "toolchain": None if tiled else {"version": "LLVM 24"},
            "toolchains": {"fusion": {"version": "LLVM 24"}} if tiled else None,
        }
        manifest_name = (
            compiler.MLIR_TILED_FFN_MANIFEST
            if tiled
            else compiler.MLIR_CUBLAS_PTX_MANIFEST
        )
        (directory / manifest_name).write_text(
            json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8"
        )
        return manifest

    def test_all_emits_legacy_and_two_offline_ptx_artifacts(self) -> None:
        output = self.root / "compiled"
        graph = object()
        cublas_program = object()
        tiled_program = object()
        cuda_program = SimpleNamespace(launches=(object(), object()))
        legacy_program = object()

        def cublas_writer(path, _program):
            return self._write_offline_manifest(Path(path), tiled=False)

        def tiled_writer(path, _program):
            return self._write_offline_manifest(Path(path), tiled=True)

        stdout = io.StringIO()
        with (
            patch.object(compiler, "build_bevformer_ffn_graph", return_value=graph),
            patch.object(compiler, "compile_cuda", return_value=cuda_program),
            patch.object(
                compiler, "write_package", side_effect=self._write_fake_package
            ),
            patch.object(compiler, "compile_mlir_cuda", return_value=legacy_program),
            patch.object(
                compiler,
                "_write_mlir_artifact",
                return_value={
                    "nvrtc_required": True,
                    "ptx_generated_offline": False,
                },
            ),
            patch.object(
                compiler,
                "compile_mlir_cublas_ptx",
                return_value=cublas_program,
            ) as compile_cublas,
            patch.object(
                compiler,
                "compile_mlir_tiled_ffn",
                return_value=tiled_program,
            ) as compile_tiled,
            patch.object(
                compiler,
                "write_mlir_cublas_ptx_artifact",
                side_effect=cublas_writer,
            ),
            patch.object(
                compiler,
                "write_mlir_tiled_ffn_artifact",
                side_effect=tiled_writer,
            ),
            redirect_stdout(stdout),
        ):
            compiler.main(
                [
                    str(self.bundle),
                    str(output),
                    "--backend",
                    "all",
                    "--mlir-opt",
                    "/opt/llvm/bin/mlir-opt",
                    "--wsl",
                    "--wsl-distribution",
                    "Ubuntu-24.04",
                    "--timeout",
                    "33",
                    "--tile",
                    "8,8,16",
                ]
            )

        toolchain = {
            "mlir_opt": "/opt/llvm/bin/mlir-opt",
            "use_wsl": True,
            "wsl_distribution": "Ubuntu-24.04",
            "timeout": 33.0,
        }
        compile_cublas.assert_called_once_with(graph, **toolchain)
        compile_tiled.assert_called_once_with(
            graph,
            tile_m=8,
            tile_n=8,
            tile_k=16,
            base_program=cublas_program,
            **toolchain,
        )
        report_path = output / "compile-report.json"
        self.assertEqual(stdout.getvalue().strip(), str(report_path.resolve()))
        self.assertFalse((output / compiler.INCOMPLETE_MARKER).exists())
        report = json.loads(report_path.read_text(encoding="utf-8"))
        self.assertEqual(
            report["selected_backends"],
            [
                "cuda-hxp",
                "mlir-cuda-sm86",
                "mlir-cublas-ptx-sm86",
                "mlir-tiled-ffn-sm86",
            ],
        )
        self.assertEqual(set(report["backends"]), {
            "cuda_hxp",
            "mlir_cuda_sm86",
            "mlir_cublas_ptx_sm86",
            "mlir_tiled_ffn_sm86",
        })
        for name in ("mlir_cublas_ptx_sm86", "mlir_tiled_ffn_sm86"):
            backend = report["backends"][name]
            self.assertTrue(backend["ptx_generated_offline"])
            self.assertFalse(backend["nvrtc_required"])
            self.assertEqual(backend["executed_custom_artifact"], "textual-ptx")
            self.assertRegex(backend["manifest"]["sha256"], r"^[0-9a-f]{64}$")
            self.assertTrue(backend["ptx_files"])
        tiled = report["backends"]["mlir_tiled_ffn_sm86"]
        self.assertTrue(tiled["base_program_reused"])
        self.assertIsNotNone(tiled["dependency_compile_ms"])
        self.assertGreaterEqual(
            tiled["compile_ms"], tiled["incremental_compile_ms"]
        )
        self.assertEqual(
            tiled["invocation_compile_ms"], tiled["incremental_compile_ms"]
        )
        self.assertIn("standalone", tiled["compile_ms_scope"])
        self.assertEqual(
            report["offline_mlir_toolchain"]["tiled_matmul_tile"], [8, 8, 16]
        )

    def test_both_retains_the_original_two_backend_selection(self) -> None:
        output = self.root / "legacy-pair"
        cuda_program = SimpleNamespace(launches=(object(),))
        with (
            patch.object(compiler, "build_bevformer_ffn_graph", return_value=object()),
            patch.object(compiler, "compile_cuda", return_value=cuda_program),
            patch.object(
                compiler, "write_package", side_effect=self._write_fake_package
            ),
            patch.object(compiler, "compile_mlir_cuda", return_value=object()),
            patch.object(
                compiler,
                "_write_mlir_artifact",
                return_value={
                    "nvrtc_required": True,
                    "ptx_generated_offline": False,
                },
            ),
            patch.object(compiler, "compile_mlir_cublas_ptx") as compile_cublas,
            patch.object(compiler, "compile_mlir_tiled_ffn") as compile_tiled,
            redirect_stdout(io.StringIO()),
        ):
            compiler.main([str(self.bundle), str(output), "--backend", "both"])

        compile_cublas.assert_not_called()
        compile_tiled.assert_not_called()
        report = json.loads(
            (output / "compile-report.json").read_text(encoding="utf-8")
        )
        self.assertEqual(
            report["selected_backends"], ["cuda-hxp", "mlir-cuda-sm86"]
        )
        self.assertEqual(set(report["backends"]), {"cuda_hxp", "mlir_cuda_sm86"})
        self.assertTrue(report["backends"]["cuda_hxp"]["nvrtc_required"])

    def test_toolchain_options_are_rejected_for_legacy_only_selection(self) -> None:
        stderr = io.StringIO()
        with (
            patch.object(compiler, "load_bundle") as load_bundle,
            redirect_stderr(stderr),
            self.assertRaises(SystemExit) as raised,
        ):
            compiler.main(
                [
                    str(self.bundle),
                    str(self.root / "output"),
                    "--backend",
                    "both",
                    "--mlir-opt",
                    "/opt/llvm/bin/mlir-opt",
                ]
            )

        self.assertEqual(raised.exception.code, 2)
        self.assertIn("MLIR toolchain options require", stderr.getvalue())
        load_bundle.assert_not_called()

    def test_existing_output_is_rejected_before_bundle_loading(self) -> None:
        output = self.root / "existing"
        output.mkdir()
        sentinel = output / "sentinel.txt"
        sentinel.write_text("keep\n", encoding="utf-8")
        stderr = io.StringIO()
        with (
            patch.object(compiler, "load_bundle") as load_bundle,
            redirect_stderr(stderr),
            self.assertRaises(SystemExit) as raised,
        ):
            compiler.main([str(self.bundle), str(output), "--backend", "both"])
        self.assertEqual(raised.exception.code, 2)
        self.assertIn("output must not already exist", stderr.getvalue())
        self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep\n")
        load_bundle.assert_not_called()

    def test_failed_compile_leaves_marker_and_no_stale_report(self) -> None:
        output = self.root / "failed"
        with (
            patch.object(
                compiler, "build_bevformer_ffn_graph", return_value=object()
            ),
            patch.object(
                compiler,
                "compile_mlir_cublas_ptx",
                side_effect=RuntimeError("lowering failed"),
            ),
            self.assertRaisesRegex(RuntimeError, "lowering failed"),
        ):
            compiler.main(
                [
                    str(self.bundle),
                    str(output),
                    "--backend",
                    "mlir-cublas-ptx-sm86",
                    "--mlir-opt",
                    "/opt/llvm/bin/mlir-opt",
                ]
            )
        self.assertTrue((output / compiler.INCOMPLETE_MARKER).is_file())
        self.assertFalse((output / "compile-report.json").exists())

    def test_wsl_distribution_requires_wsl(self) -> None:
        stderr = io.StringIO()
        with redirect_stderr(stderr), self.assertRaises(SystemExit) as raised:
            compiler.main(
                [
                    str(self.bundle),
                    str(self.root / "output"),
                    "--backend",
                    "mlir-cublas-ptx-sm86",
                    "--wsl-distribution",
                    "Ubuntu-24.04",
                ]
            )
        self.assertEqual(raised.exception.code, 2)
        self.assertIn("--wsl-distribution requires --wsl", stderr.getvalue())

    def test_offline_ptx_rejects_nonproduction_uniad_contract(self) -> None:
        small_bundle = self.root / "small.uniadffn"
        contract = UniADFFNContract(
            layer=0,
            batch=1,
            bev_h=50,
            bev_w=50,
            embed_dim=256,
            hidden_dim=512,
        )
        write_bundle(
            small_bundle,
            contract,
            {
                name: np.zeros(shape, dtype=np.float32)
                for name, shape in contract.weight_shapes.items()
            },
        )
        stderr = io.StringIO()
        with (
            patch.object(compiler, "build_bevformer_ffn_graph") as build_graph,
            redirect_stderr(stderr),
            self.assertRaises(SystemExit) as raised,
        ):
            compiler.main(
                [
                    str(small_bundle),
                    str(self.root / "offline-output"),
                    "--backend",
                    "mlir-cublas-ptx-sm86",
                    "--mlir-opt",
                    "/opt/llvm/bin/mlir-opt",
                ]
            )
        self.assertEqual(raised.exception.code, 2)
        self.assertIn("production UniAD profile", stderr.getvalue())
        build_graph.assert_not_called()

        parser = compiler._build_parser()
        args = parser.parse_args(
            [str(small_bundle), str(self.root / "legacy-output"), "--backend", "both"]
        )
        compiler._validate_offline_profile(parser, args, contract)


if __name__ == "__main__":
    unittest.main()
