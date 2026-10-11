from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
import hashlib
from io import StringIO
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from gac.backends.mlir_ptx import (
    FC1_SYMBOL,
    FC2_SYMBOL,
    FINAL_SYMBOL,
    PTX_ENTRY_SYMBOLS,
    TARGET,
    assemble_mlir_ptx_program,
    compile_mlir_ptx,
    validate_mlir_ptx_program,
)
from gac.bevformer_ffn import (
    BEVFormerFFNConfig,
    build_bevformer_ffn_graph,
    ffn_weight_shapes,
)
from gac.errors import BackendError
from gac.cli import main as cli_main
from gac.mlir_ptx_artifact import (
    load_mlir_ptx_artifact,
    write_mlir_ptx_artifact,
)
from gac.mlir_toolchain import MlirPtxArtifact
from gac.runtime import reference_run


def _weights(config: BEVFormerFFNConfig) -> dict[str, np.ndarray]:
    return {
        name: np.zeros(shape, dtype=np.float32)
        for name, shape in ffn_weight_shapes(config).items()
    }


def _production_graph():
    config = BEVFormerFFNConfig()
    return build_bevformer_ffn_graph(
        _weights(config), config, name="bevformer_production_ffn_mlir_ptx"
    )


def _mlir_string(value: str) -> str:
    encoded: list[str] = []
    for byte in value.encode("utf-8"):
        if 0x20 <= byte <= 0x7E and byte not in (0x22, 0x5C):
            encoded.append(chr(byte))
        else:
            encoded.append(f"\\{byte:02X}")
    return "".join(encoded)


def _fake_artifact(
    source: str, *, parameter_count_overrides: dict[str, int] | None = None
) -> MlirPtxArtifact:
    parameter_counts = {
        FC1_SYMBOL: 4,
        FC2_SYMBOL: 3,
        FINAL_SYMBOL: 6,
    }
    if parameter_count_overrides:
        parameter_counts.update(parameter_count_overrides)
    entries = []
    for symbol in PTX_ENTRY_SYMBOLS:
        parameters = ",\n".join(
            f"  .param .u64 .ptr .global {symbol}_param_{index}"
            for index in range(parameter_counts[symbol])
        )
        entries.append(
            f".visible .entry {symbol}(\n{parameters}\n)\n"
            ".maxntid 256, 1, 1\n"
            "{ ret; }"
        )
    entry_source = "\n".join(entries)
    ptx = f".version 7.1\n.target sm_86\n.address_size 64\n{entry_source}\n"
    binary_mlir = (
        "module {\n"
        "  gpu.binary @gac_bevformer_ffn_sm86 "
        "[#gpu.object<#nvvm.target<chip = \"sm_86\">, "
        f'properties = {{assembly = "{_mlir_string(ptx)}"}}, "PTX">]\n'
        "}\n"
    )
    return MlirPtxArtifact(
        binary_mlir=binary_mlir,
        ptx=ptx,
        source_sha256=hashlib.sha256(source.encode("utf-8")).hexdigest(),
        command=("mlir-opt", "<input>", "--gpu-lower-to-nvvm-pipeline=..."),
        mlir_opt_version="LLVM 24.0.0git (Optimized build)",
        mlir_opt_source_revision="765392a6b4498f1dc57233f03a31b87462fdef01",
        target_chip="sm_86",
        target_features="+ptx71",
        ptx_version="7.1",
        entries=PTX_ENTRY_SYMBOLS,
    )


def _fake_lowering(source: str, **_kwargs) -> MlirPtxArtifact:
    return _fake_artifact(source)


def _invoke_cli(arguments: list[str]) -> tuple[int, str, str]:
    stdout = StringIO()
    stderr = StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        code = cli_main(arguments)
    return code, stdout.getvalue(), stderr.getvalue()


class MlirPtxBackendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.graph = _production_graph()
        with patch(
            "gac.mlir_toolchain.lower_gpu_mlir_to_ptx",
            side_effect=_fake_lowering,
        ) as lower:
            cls.program = compile_mlir_ptx(
                cls.graph,
                command=("mock-mlir-opt",),
                timeout=17,
            )
        cls.lower_call = lower.call_args
        cls.artifact = _fake_artifact(cls.program.gpu_source)

    def test_emits_complete_three_kernel_gpu_module(self):
        source = self.program.gpu_source
        self.assertIn("module attributes {gpu.container_module}", source)
        self.assertEqual(source.count("gpu.func @"), 3)
        for symbol in PTX_ENTRY_SYMBOLS:
            self.assertIn(f"gpu.func @{symbol}(", source)
        self.assertIn("scf.for", source)
        self.assertIn("gpu.all_reduce add", source)
        self.assertIn("nvvm.rsqrt", source)
        self.assertNotIn("func.call", source)
        self.assertNotIn("cublas", source.lower())
        self.assertNotIn("nvrtc", source.lower())

        _, kwargs = self.lower_call
        self.assertEqual(tuple(kwargs["expected_entries"]), PTX_ENTRY_SYMBOLS)
        self.assertEqual(tuple(kwargs["command"]), ("mock-mlir-opt",))
        self.assertEqual(kwargs["timeout"], 17)

    def test_launch_plan_covers_every_ffn_operation_with_ptx(self):
        launches = self.program.launches
        self.assertEqual(
            [launch.symbol for launch in launches],
            [FC1_SYMBOL, FC2_SYMBOL, FINAL_SYMBOL],
        )
        self.assertEqual([launch.block for launch in launches], [(256, 1, 1)] * 3)
        self.assertEqual(launches[0].grid, (5000, 1, 1))
        self.assertEqual(launches[1].grid, (2500, 1, 1))
        self.assertEqual(launches[2].grid, (2500, 1, 1))
        self.assertEqual(len(launches[0].inputs), 3)
        self.assertEqual(len(launches[1].inputs), 2)
        self.assertEqual(len(launches[2].inputs), 5)
        self.assertIs(self.program.kernels, self.program.launches)

    def test_description_states_executed_ptx_boundary(self):
        description = self.program.describe()
        self.assertEqual(description["compiler_backend"], TARGET)
        self.assertEqual(description["lowering_stage"], "ptx")
        self.assertTrue(description["mlir_source_lowered"])
        self.assertFalse(description["mlir_source_executed"])
        self.assertEqual(description["executed_artifact"], "textual-ptx")
        self.assertEqual(
            description["mlir_source_role"], "lowering-input-for-executed-ptx"
        )
        self.assertEqual(description["dispatch_count"], 3)
        self.assertEqual(
            description["matmul_lowering"], "mlir-gpu-serial-f32"
        )
        self.assertEqual(description["toolchain"]["chip"], "sm_86")
        self.assertEqual(
            description["toolchain"]["llvm_source_revision"],
            "765392a6b4498f1dc57233f03a31b87462fdef01",
        )
        self.assertEqual(len(description["integrity"]["ptx_sha256"]), 64)

    def test_program_validates_without_reinvoking_mlir(self):
        with patch(
            "gac.mlir_toolchain.lower_gpu_mlir_to_ptx",
            side_effect=AssertionError("validation must not invoke the compiler"),
            create=True,
        ):
            validate_mlir_ptx_program(self.program)

    def test_offline_artifact_can_be_assembled_without_invoking_mlir(self):
        with patch(
            "gac.mlir_toolchain.lower_gpu_mlir_to_ptx",
            side_effect=AssertionError("assembly must not invoke the compiler"),
        ):
            rebuilt = assemble_mlir_ptx_program(self.graph, self.artifact)
        self.assertEqual(rebuilt.integrity_sha256, self.program.integrity_sha256)
        self.assertEqual(rebuilt.ptx, self.program.ptx)
        self.assertEqual(rebuilt.gpu_source, self.program.gpu_source)

    def test_offline_artifact_is_bound_to_gpu_source_and_binary_ptx(self):
        with self.assertRaisesRegex(BackendError, "not produced from this"):
            assemble_mlir_ptx_program(
                self.graph,
                replace(self.artifact, source_sha256="0" * 64),
            )
        with self.assertRaisesRegex(BackendError, "do not match"):
            assemble_mlir_ptx_program(
                self.graph,
                replace(self.artifact, ptx=self.artifact.ptx + "// altered\n"),
            )

    def test_offline_artifact_requires_exact_pointer_and_launch_abi(self):
        wrong_count = _fake_artifact(
            self.program.gpu_source,
            parameter_count_overrides={FC1_SYMBOL: 3},
        )
        with self.assertRaisesRegex(BackendError, "requires 4 global u64 pointer"):
            assemble_mlir_ptx_program(self.graph, wrong_count)

        wrong_maxntid_ptx = self.artifact.ptx.replace(
            ".maxntid 256, 1, 1", ".maxntid 128, 1, 1", 1
        )
        wrong_maxntid_binary = self.artifact.binary_mlir.replace(
            _mlir_string(self.artifact.ptx), _mlir_string(wrong_maxntid_ptx)
        )
        with self.assertRaisesRegex(BackendError, "requires \\.maxntid 256"):
            assemble_mlir_ptx_program(
                self.graph,
                replace(
                    self.artifact,
                    binary_mlir=wrong_maxntid_binary,
                    ptx=wrong_maxntid_ptx,
                ),
            )

    def test_integrity_rejects_modified_ptx_and_gpu_mlir(self):
        with self.assertRaisesRegex(BackendError, "binary MLIR and PTX do not match"):
            validate_mlir_ptx_program(
                replace(self.program, ptx=self.program.ptx + "// modified\n")
            )
        with self.assertRaisesRegex(BackendError, "GPU MLIR"):
            validate_mlir_ptx_program(
                replace(
                    self.program,
                    gpu_source=self.program.gpu_source + "// modified\n",
                )
            )

    def test_rejects_non_production_shape_before_toolchain(self):
        config = BEVFormerFFNConfig(bev_h=1, bev_w=2, embed_dim=4)
        graph = build_bevformer_ffn_graph(_weights(config), config)
        with patch(
            "gac.mlir_toolchain.lower_gpu_mlir_to_ptx",
            side_effect=AssertionError("toolchain must not run"),
            create=True,
        ) as lower:
            with self.assertRaisesRegex(BackendError, "requires query shape"):
                compile_mlir_ptx(graph, command=("mock-mlir-opt",))
        lower.assert_not_called()

    def test_weight_free_directory_round_trip_is_bound_to_the_graph(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "artifact"
            manifest = write_mlir_ptx_artifact(root, self.program)
            self.assertEqual(
                set(manifest["files"]), {"gpu_mlir", "binary_mlir", "ptx"}
            )
            self.assertFalse((root / "tensor.mlir").exists())
            rebuilt = load_mlir_ptx_artifact(root, self.graph)
            self.assertEqual(rebuilt.integrity_sha256, self.program.integrity_sha256)

            other_graph = self.graph.clone()
            other_graph.constants["fc1_bias"][0] += np.float32(1.0)
            with self.assertRaisesRegex(BackendError, "not bound to this graph"):
                load_mlir_ptx_artifact(root, other_graph)

            manifest_path = root / "manifest.json"
            original_manifest = manifest_path.read_text(encoding="utf-8")
            manifest = json.loads(original_manifest)
            manifest["executed_artifact"] = "cubin"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(BackendError, "metadata does not match"):
                load_mlir_ptx_artifact(root, self.graph)
            manifest_path.write_text(original_manifest, encoding="utf-8")

            (root / "kernels.ptx").write_text("modified\n", encoding="utf-8")
            with self.assertRaisesRegex(BackendError, "(?:size|SHA-256) does not match"):
                load_mlir_ptx_artifact(root, self.graph)

    def test_cli_publishes_safe_directory_and_rejects_report_collision(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "artifact"
            colliding_report = artifact / "kernels.ptx"
            with (
                patch("gac.cli.load_graph") as load_mock,
                patch("gac.cli.compile_mlir_ptx") as compile_mock,
            ):
                code, stdout, stderr = _invoke_cli(
                    [
                        "mlir-ptx-compile",
                        "model.json",
                        "-o",
                        str(artifact),
                        "--report",
                        str(colliding_report),
                    ]
                )
            self.assertEqual(code, 1)
            self.assertEqual(stdout, "")
            self.assertIn("outside the artifact directory", stderr)
            load_mock.assert_not_called()
            compile_mock.assert_not_called()

            report_path = root / "compile-report.json"
            with (
                patch("gac.cli.load_graph", return_value=self.graph),
                patch("gac.cli.compile_mlir_ptx", return_value=self.program),
            ):
                code, stdout, stderr = _invoke_cli(
                    [
                        "mlir-ptx-compile",
                        "model.json",
                        "-o",
                        str(artifact),
                        "--report",
                        str(report_path),
                    ]
                )
            self.assertEqual(code, 0, stderr)
            self.assertEqual(stderr, "")
            report = json.loads(stdout)
            self.assertEqual(report["artifact_directory"], str(artifact.resolve()))
            self.assertTrue(report_path.is_file())
            self.assertFalse((artifact / "tensor.mlir").exists())
            manifest = json.loads(
                (artifact / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                set(manifest["files"]), {"gpu_mlir", "binary_mlir", "ptx"}
            )

    def test_cli_run_loads_offline_artifact_without_invoking_mlir(self):
        program = self.program

        class FakeSession:
            def __init__(self, selected, *, device_id):
                self.program = selected
                self.info = {
                    "name": "fake SM86",
                    "compute_capability": [8, 6],
                    "compiler_backend": TARGET,
                }
                self.setup_ms = 2.0
                self.kernel_info = [launch.to_dict() for launch in selected.launches]

            def __enter__(self):
                return self

            def __exit__(self, *_):
                return None

            def run(self, inputs):
                return SimpleNamespace(
                    outputs=reference_run(self.program.reference_graph, inputs),
                    wall_ms=1.5,
                    gpu_stream_ms=0.5,
                )

        with (
            patch("gac.cli.load_graph", return_value=self.graph),
            patch(
                "gac.cli.load_mlir_ptx_artifact", return_value=program
            ) as load_artifact,
            patch("gac.cli.compile_mlir_ptx") as compile_mock,
            patch("gac.cli.MlirPtxSession", FakeSession),
        ):
            code, stdout, stderr = _invoke_cli(
                [
                    "run",
                    "model.json",
                    "--backend",
                    TARGET,
                    "--mlir-ptx-artifact",
                    "artifact",
                    "--warmup",
                    "0",
                    "--repeat",
                    "1",
                ]
            )
        self.assertEqual(code, 0, stderr)
        self.assertEqual(stderr, "")
        load_artifact.assert_called_once_with("artifact", self.graph)
        compile_mock.assert_not_called()
        report = json.loads(stdout)
        self.assertEqual(report["backend"], TARGET)
        self.assertEqual(
            report["program_preparation"], "verified-precompiled-mlir-ptx-artifact"
        )

    def test_cli_run_rejects_reports_that_can_overwrite_inputs_or_artifacts(self):
        cases = (
            (["--report", "model.json"], "must not overwrite the model"),
            (
                ["--inputs", "inputs.npz", "--report", "inputs.npz"],
                "must not overwrite the input archive",
            ),
            (
                [
                    "--mlir-ptx-artifact",
                    "artifact",
                    "--report",
                    str(Path("artifact") / "kernels.ptx"),
                ],
                "outside the MLIR PTX artifact directory",
            ),
            (
                ["--outputs", "result.npz", "--report", "result.npz"],
                "must name different files",
            ),
            (
                ["--inputs", "inputs.npz", "--outputs", "inputs.npz"],
                "must not overwrite the input archive",
            ),
            (
                [
                    "--mlir-ptx-artifact",
                    "artifact",
                    "--outputs",
                    str(Path("artifact") / "outputs"),
                ],
                "outside the MLIR PTX artifact directory",
            ),
        )
        for extra, expected_error in cases:
            with self.subTest(extra=extra), patch("gac.cli.load_graph") as load_mock:
                code, stdout, stderr = _invoke_cli(
                    ["run", "model.json", "--backend", TARGET, *extra]
                )
            self.assertEqual(code, 1)
            self.assertEqual(stdout, "")
            self.assertIn(expected_error, stderr)
            load_mock.assert_not_called()

    def test_benchmark_rejects_output_inside_loaded_or_emitted_artifact(self):
        script = Path(__file__).resolve().parents[1] / "tools" / "benchmark_mlir_ptx_ffn.py"
        cases = (
            [
                "--compile-only",
                "--emit-artifact-dir",
                "artifact",
                "--output",
                str(Path("artifact") / "kernels.ptx"),
            ],
            [
                "--artifact-dir",
                "artifact",
                "--output",
                str(Path("artifact") / "manifest.json"),
            ],
        )
        for arguments in cases:
            with self.subTest(arguments=arguments):
                completed = subprocess.run(
                    [sys.executable, str(script), *arguments],
                    cwd=script.parents[1],
                    check=False,
                    capture_output=True,
                    text=True,
                )
            self.assertEqual(completed.returncode, 2)
            self.assertIn("--output must be outside", completed.stderr)

    def test_cli_run_rejects_hardlink_alias_of_artifact_member(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "artifact"
            artifact.mkdir()
            ptx = artifact / "kernels.ptx"
            ptx.write_text("device code", encoding="utf-8")
            alias = root / "outside.npz"
            try:
                alias.hardlink_to(ptx)
            except OSError as exc:
                self.skipTest(f"hard links are unavailable: {exc}")

            with patch("gac.cli.load_graph") as load_mock:
                code, stdout, stderr = _invoke_cli(
                    [
                        "run",
                        "model.json",
                        "--backend",
                        TARGET,
                        "--mlir-ptx-artifact",
                        str(artifact),
                        "--outputs",
                        str(alias),
                    ]
                )
            self.assertEqual(code, 1)
            self.assertEqual(stdout, "")
            self.assertIn("outside the MLIR PTX artifact directory", stderr)
            self.assertEqual(ptx.read_text(encoding="utf-8"), "device code")
            load_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
