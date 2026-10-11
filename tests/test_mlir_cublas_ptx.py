"""Contracts for the cuBLAS plus official MLIR PTX FFN backend."""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from gac.backends.mlir_cublas_ptx import (
    BIAS_RELU_SYMBOL,
    FINAL_SYMBOL,
    PTX_ENTRY_SYMBOLS,
    TARGET,
    assemble_mlir_cublas_ptx_program,
    compile_mlir_cublas_ptx,
    validate_mlir_cublas_ptx_program,
)
from gac.bevformer_ffn import (
    BEVFormerFFNConfig,
    build_bevformer_ffn_graph,
    ffn_weight_shapes,
)
from gac.errors import BackendError, DeviceUnavailableError, ValidationError
from gac.cli import main as cli_main
from gac.mlir_cublas_ptx_runtime import MlirCublasPtxSession
from gac.mlir_toolchain import MlirPtxArtifact


def _graph(config: BEVFormerFFNConfig | None = None):
    config = config or BEVFormerFFNConfig()
    weights = {
        name: np.zeros(shape, dtype=np.float32)
        for name, shape in ffn_weight_shapes(config).items()
    }
    if "norm_weight" in weights:
        weights["norm_weight"].fill(1.0)
    return build_bevformer_ffn_graph(
        weights, config, name="bevformer_production_cublas_ptx"
    )


def _mlir_string(value: str) -> str:
    return "".join(f"\\{byte:02X}" for byte in value.encode("utf-8"))


def _artifact(
    source: str,
    *,
    parameter_counts: tuple[int, int] = (2, 6),
    maxntid: str = "256, 1, 1",
) -> MlirPtxArtifact:
    entries = []
    for symbol, count in zip(PTX_ENTRY_SYMBOLS, parameter_counts, strict=True):
        parameters = ",\n".join(
            f"  .param .u64 .ptr .global {symbol}_param_{index}"
            for index in range(count)
        )
        entries.append(
            f".visible .entry {symbol}(\n{parameters}\n)\n"
            f".maxntid {maxntid}\n"
            "{ ret; }"
        )
    ptx = (
        ".version 7.1\n"
        ".target sm_86\n"
        ".address_size 64\n"
        + "\n".join(entries)
        + "\n"
    )
    binary_mlir = (
        "module {\n"
        "  gpu.binary @gac_bevformer_ffn_fusions_sm86 "
        "[#gpu.object<#nvvm.target<chip = \"sm_86\">, "
        f'properties = {{assembly = "{_mlir_string(ptx)}"}}, "PTX">]\n'
        "}\n"
    )
    return MlirPtxArtifact(
        binary_mlir=binary_mlir,
        ptx=ptx,
        source_sha256=hashlib.sha256(source.encode("utf-8")).hexdigest(),
        command=("mlir-opt", "--gpu-lower-to-nvvm-pipeline=..."),
        mlir_opt_version="LLVM 24.0.0git (Optimized build)",
        mlir_opt_source_revision="a" * 40,
        target_chip="sm_86",
        target_features="+ptx71",
        ptx_version="7.1",
        entries=PTX_ENTRY_SYMBOLS,
    )


def _fake_lowering(source: str, **_kwargs) -> MlirPtxArtifact:
    return _artifact(source)


class MlirCublasPtxBackendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.graph = _graph()
        with patch(
            "gac.mlir_toolchain.lower_gpu_mlir_to_ptx",
            side_effect=_fake_lowering,
        ) as lower:
            cls.program = compile_mlir_cublas_ptx(
                cls.graph, command=("mock-mlir-opt",), timeout=19
            )
        cls.lower_call = lower.call_args

    def test_emits_two_fusion_gpu_functions_and_explicit_four_call_plan(self):
        source = self.program.gpu_source
        self.assertEqual(source.count("gpu.func @"), 2)
        self.assertIn(f"gpu.func @{BIAS_RELU_SYMBOL}(", source)
        self.assertIn(f"gpu.func @{FINAL_SYMBOL}(", source)
        self.assertIn("gpu.all_reduce add", source)
        self.assertIn("nvvm.rsqrt", source)
        self.assertNotIn("scf.for", source)
        self.assertNotIn("func.call", source)
        self.assertNotIn("nvrtc", source.lower())

        self.assertEqual(
            [call.implementation for call in self.program.calls],
            ["cupy-cublas", "mlir-ptx", "cupy-cublas", "mlir-ptx"],
        )
        self.assertEqual(self.program.calls[1].aliases_input, 0)
        self.assertEqual(self.program.calls[1].grid, (5000, 1, 1))
        self.assertEqual(self.program.calls[3].grid, (2500, 1, 1))
        self.assertEqual([call.block for call in self.program.calls[1::2]], [(256, 1, 1)] * 2)

        _, kwargs = self.lower_call
        self.assertEqual(tuple(kwargs["expected_entries"]), PTX_ENTRY_SYMBOLS)
        self.assertEqual(tuple(kwargs["command"]), ("mock-mlir-opt",))
        self.assertEqual(kwargs["timeout"], 19)

    def test_description_reports_cublas_ptx_boundary_without_nvrtc(self):
        description = self.program.describe()
        self.assertEqual(description["compiler_backend"], TARGET)
        self.assertEqual(description["matmul_lowering"], "cupy-cublas")
        self.assertEqual(description["fusion_lowering"], "mlir-gpu-nvvm-ptx")
        self.assertEqual(description["cublas_dispatch_count"], 2)
        self.assertEqual(description["ptx_dispatch_count"], 2)
        self.assertFalse(description["nvrtc_required"])
        self.assertNotIn("NVRTC", description["runtime_requirements"])

    def test_validation_and_offline_assembly_do_not_reinvoke_mlir(self):
        artifact = _artifact(self.program.gpu_source)
        with patch(
            "gac.mlir_toolchain.lower_gpu_mlir_to_ptx",
            side_effect=AssertionError("must not invoke compiler"),
        ):
            validate_mlir_cublas_ptx_program(self.program)
            rebuilt = assemble_mlir_cublas_ptx_program(self.graph, artifact)
        self.assertEqual(rebuilt.integrity_sha256, self.program.integrity_sha256)

    def test_tampering_and_wrong_ptx_abi_fail_closed(self):
        for invalid in (True, 2.0):
            with self.subTest(abi_version=invalid), self.assertRaisesRegex(
                BackendError, "execution metadata"
            ):
                validate_mlir_cublas_ptx_program(
                    replace(self.program, abi_version=invalid)
                )
        with self.assertRaisesRegex(BackendError, "integrity"):
            validate_mlir_cublas_ptx_program(
                replace(self.program, integrity_sha256="0" * 64)
            )
        with self.assertRaisesRegex(BackendError, "exactly 2 bare global pointers"):
            assemble_mlir_cublas_ptx_program(
                self.graph,
                _artifact(self.program.gpu_source, parameter_counts=(3, 6)),
            )
        with self.assertRaisesRegex(BackendError, "maxntid"):
            assemble_mlir_cublas_ptx_program(
                self.graph,
                _artifact(self.program.gpu_source, maxntid="128, 1, 1"),
            )

    def test_non_production_shape_is_rejected(self):
        with self.assertRaisesRegex(BackendError, "query shape"):
            compile_mlir_cublas_ptx(
                _graph(BEVFormerFFNConfig(bev_h=2, bev_w=3, embed_dim=32)),
                command=("must-not-run",),
            )


class _FakeContext:
    def __enter__(self):
        return self

    def __exit__(self, *_):
        return None


class _FakeStream(_FakeContext):
    def __init__(self):
        self.synchronize_calls = 0

    def synchronize(self):
        self.synchronize_calls += 1


class _FakeArray:
    _next_pointer = 0x1000

    def __init__(self, shape, dtype, value=None):
        self.shape = tuple(int(dimension) for dimension in shape)
        self.dtype = np.dtype(dtype)
        self.flags = SimpleNamespace(c_contiguous=True)
        self.value = np.empty(self.shape, dtype=self.dtype)
        if value is not None:
            self.value[...] = np.asarray(value, dtype=self.dtype)
        self.data = SimpleNamespace(ptr=_FakeArray._next_pointer)
        _FakeArray._next_pointer += max(0x100, self.value.nbytes)

    def set(self, value, *, stream):
        self.value[...] = np.asarray(value, dtype=self.dtype)


class _FakeKernel:
    def __init__(self, symbol):
        self.symbol = symbol
        self.attributes = {"max_threads_per_block": 1024}
        self.calls = []

    def __call__(self, grid, block, arguments, *, stream):
        self.calls.append((grid, block, arguments, stream))


class _FakeModule:
    def __init__(self, path):
        self.path = Path(path)
        self.ptx_at_load = self.path.read_text(encoding="utf-8")
        self.kernels = {}

    def get_function(self, symbol):
        kernel = _FakeKernel(symbol)
        self.kernels[symbol] = kernel
        return kernel


class _FakeCupy:
    def __init__(self):
        self.device = _FakeContext()
        self.stream = _FakeStream()
        self.modules = []
        self.raw_module_arguments = []
        self.matmul_calls = []
        self.cuda = SimpleNamespace(
            Device=lambda _: self.device,
            Stream=lambda **_: self.stream,
        )

    def RawModule(self, **arguments):
        self.raw_module_arguments.append(arguments)
        module = _FakeModule(arguments["path"])
        self.modules.append(module)
        return module

    @staticmethod
    def array(value, *, dtype, order, copy):
        return _FakeArray(np.asarray(value).shape, dtype, value)

    @staticmethod
    def empty(shape, *, dtype, order):
        return _FakeArray(shape, dtype)

    def matmul(self, left, right, *, out):
        self.matmul_calls.append((left, right, out))


def _device_info(capability=(8, 6)):
    return {
        "device_id": 0,
        "name": "fake-sm86",
        "compute_capability": list(capability),
    }


class MlirCublasPtxRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        graph = _graph()
        with patch(
            "gac.mlir_toolchain.lower_gpu_mlir_to_ptx",
            side_effect=_fake_lowering,
        ):
            cls.program = compile_mlir_cublas_ptx(graph)

    def test_non_program_and_tampering_are_rejected_before_cuda(self):
        with patch(
            "gac.mlir_cublas_ptx_runtime.cuda_device_info"
        ) as device_probe:
            with self.assertRaisesRegex(ValidationError, "MlirCublasPtxProgram"):
                MlirCublasPtxSession(object())
        device_probe.assert_not_called()

        with patch(
            "gac.mlir_cublas_ptx_runtime.cuda_device_info"
        ) as device_probe:
            with self.assertRaisesRegex(BackendError, "integrity"):
                MlirCublasPtxSession(
                    replace(self.program, integrity_sha256="0" * 64)
                )
        device_probe.assert_not_called()

    def test_wrong_device_is_rejected_before_cupy_or_ptx_loading(self):
        fake = _FakeCupy()
        with (
            patch(
                "gac.mlir_cublas_ptx_runtime.cuda_device_info",
                return_value=_device_info((8, 9)),
            ),
            patch(
                "gac.mlir_cublas_ptx_runtime._cupy", return_value=fake
            ) as cupy_probe,
        ):
            with self.assertRaisesRegex(DeviceUnavailableError, "8.6"):
                MlirCublasPtxSession(self.program)
        cupy_probe.assert_not_called()
        self.assertFalse(fake.raw_module_arguments)

    def test_runtime_loads_ptx_path_and_dispatches_two_cublas_two_ptx(self):
        fake = _FakeCupy()
        with (
            patch(
                "gac.mlir_cublas_ptx_runtime.cuda_device_info",
                return_value=_device_info(),
            ),
            patch("gac.mlir_cublas_ptx_runtime._cupy", return_value=fake),
        ):
            session = MlirCublasPtxSession(self.program)

        self.assertEqual(len(fake.raw_module_arguments), 1)
        self.assertEqual(set(fake.raw_module_arguments[0]), {"path"})
        self.assertEqual(fake.modules[0].ptx_at_load, self.program.ptx)
        self.assertEqual(set(session._kernels), set(PTX_ENTRY_SYMBOLS))
        self.assertEqual(session.info["matmul_backend"]["call_count"], 2)
        self.assertEqual(session.info["fusion_backend"]["launch_count"], 2)
        self.assertEqual(session.info["fusion_backend"]["nvrtc_launch_count"], 0)

        session._execute_calls()
        self.assertEqual(len(fake.matmul_calls), 2)
        self.assertEqual(sum(len(kernel.calls) for kernel in session._kernels.values()), 2)
        ptx_path = Path(fake.raw_module_arguments[0]["path"])
        self.assertTrue(ptx_path.exists())
        session.close()
        self.assertFalse(ptx_path.exists())


class MlirCublasPtxCliTests(unittest.TestCase):
    def test_run_selects_cublas_ptx_backend_and_forwards_toolchain_options(self):
        graph = _graph()
        with patch(
            "gac.mlir_toolchain.lower_gpu_mlir_to_ptx", side_effect=_fake_lowering
        ):
            program = compile_mlir_cublas_ptx(graph)

        class FakeSession:
            def __init__(self, selected, *, device_id):
                self.program = selected
                self.device_id = device_id
                self.info = {
                    "name": "fake SM86",
                    "compute_capability": [8, 6],
                    "compiler_backend": TARGET,
                }
                self.setup_ms = 2.5
                self.kernel_info = [call.to_dict() for call in selected.calls]
                output_spec = selected.graph.value_specs[selected.graph.outputs[0]]
                self.output = np.zeros(output_spec.shape, dtype=output_spec.dtype)

            def __enter__(self):
                return self

            def __exit__(self, *_):
                return None

            def run(self, _inputs):
                return SimpleNamespace(
                    outputs={"output": self.output.copy()},
                    wall_ms=1.5,
                    gpu_stream_ms=0.22,
                )

        stdout = io.StringIO()
        stderr = io.StringIO()
        with (
            patch("gac.cli.load_graph", return_value=graph),
            patch(
                "gac.cli.compile_mlir_cublas_ptx", return_value=program
            ) as compile_mock,
            patch("gac.cli.MlirCublasPtxSession", FakeSession),
            patch("gac.cli.compile_cuda") as legacy_compile,
            redirect_stdout(stdout),
            redirect_stderr(stderr),
        ):
            code = cli_main(
                [
                    "run",
                    "ffn.json",
                    "--backend",
                    TARGET,
                    "--device",
                    "2",
                    "--matmul-backend",
                    "cupy",
                    "--mlir-opt",
                    "/opt/mlir-opt",
                    "--wsl",
                    "--wsl-distribution",
                    "Ubuntu-24.04",
                    "--mlir-timeout",
                    "33",
                    "--warmup",
                    "1",
                    "--repeat",
                    "2",
                ]
            )

        self.assertEqual(code, 0, stderr.getvalue())
        self.assertEqual(stderr.getvalue(), "")
        compile_mock.assert_called_once_with(
            graph,
            mlir_opt="/opt/mlir-opt",
            use_wsl=True,
            wsl_distribution="Ubuntu-24.04",
            timeout=33.0,
        )
        legacy_compile.assert_not_called()
        report = json.loads(stdout.getvalue())
        self.assertEqual(report["backend"], TARGET)
        self.assertEqual(
            report["program_preparation"], "official-mlir-offline-compilation"
        )
        self.assertEqual(report["source_generation_ms"], 0.0)
        self.assertEqual(report["device"]["compute_capability"], [8, 6])
        self.assertEqual(report["gpu_stream_ms"]["median"], 0.22)
        self.assertEqual(report["wall_ms"]["median"], 1.5)
        self.assertEqual(len(report["runtime_calls"]), 4)
        self.assertEqual(report["lowering"]["ptx_dispatch_count"], 2)
        self.assertFalse(report["lowering"]["nvrtc_required"])

    def test_run_rejects_mlir_toolchain_options_for_non_ptx_backend(self):
        stderr = io.StringIO()
        with (
            patch("gac.cli.load_graph") as load_graph,
            redirect_stderr(stderr),
        ):
            code = cli_main(
                [
                    "run",
                    "ffn.json",
                    "--backend",
                    "mlir-cuda-sm86",
                    "--mlir-opt",
                    "/opt/mlir-opt",
                ]
            )

        self.assertEqual(code, 1)
        self.assertIn("MLIR toolchain options require", stderr.getvalue())
        load_graph.assert_not_called()


if __name__ == "__main__":
    unittest.main()
