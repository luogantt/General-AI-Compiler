from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path
from types import SimpleNamespace
import json
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from gac.bevformer_ffn import (
    BEVFormerFFNConfig,
    build_bevformer_ffn_graph,
    ffn_weight_shapes,
)
from gac.errors import BackendError, DeviceUnavailableError, ValidationError
from gac.backends.mlir_cublas_ptx import TARGET as CUBLAS_PTX_TARGET
from gac.backends.mlir_cuda import TARGET as MLIR_CUDA_TARGET
from gac.mlir_cublas_ptx_benchmark import (
    FfnSessionAutotuneRunner,
    executable_ffn_candidates,
    tiled_ffn_candidate,
)
from gac.mlir_tiled_ffn import (
    IMPLEMENTATION,
    TARGET,
    compile_mlir_tiled_ffn,
    validate_mlir_tiled_ffn_program,
)
from gac.mlir_tiled_ffn_runtime import MlirTiledFfnSession
from gac.mlir_tiled_ffn_artifact import (
    ARTIFACT_SCHEMA_VERSION,
    LEGACY_ARTIFACT_SCHEMA_VERSION,
    load_mlir_tiled_ffn_artifact,
    write_mlir_tiled_ffn_artifact,
)
from gac.mlir_toolchain import MlirPtxArtifact


def _graph(rows=2500):
    if rows == 2500:
        config = BEVFormerFFNConfig()
    elif rows == 40000:
        config = BEVFormerFFNConfig(bev_h=200, bev_w=200)
    else:
        config = BEVFormerFFNConfig(bev_h=1, bev_w=rows)
    weights = {
        name: np.zeros(shape, dtype=np.float32)
        for name, shape in ffn_weight_shapes(config).items()
    }
    weights["norm_weight"].fill(1.0)
    return build_bevformer_ffn_graph(weights, config, name="tiled_ffn_test")


def _mlir_string(value: str) -> str:
    return "".join(f"\\{byte:02X}" for byte in value.encode("utf-8"))


def _artifact(source: str, entries: tuple[str, ...]) -> MlirPtxArtifact:
    bodies: list[str] = []
    for symbol in entries:
        if symbol == "gac_bias_relu_inplace_f32":
            count, block = 2, (256, 1, 1)
            shared = ""
        elif symbol == "gac_bias_residual_layer_norm_f32":
            count, block = 6, (256, 1, 1)
            shared = ""
        else:
            fused = "fc1" in symbol
            count = 4 if fused else 3
            if "known_block_size = array<i32: 8, 8, 1>" in source:
                block = (8, 8, 1)
                first_bytes = 8 * 16 * 4
                second_bytes = 16 * 8 * 4
            else:
                block = (16, 16, 1)
                first_bytes = 16 * 16 * 4
                second_bytes = 16 * 16 * 4
            shared = (
                f"  .shared .align 4 .b8 a[{first_bytes}];\n"
                f"  .shared .align 4 .b8 b[{second_bytes}];\n"
                "  ld.shared.b32 %r1, [%rd1];\n"
                "  st.shared.b32 [%rd2], %r1;\n"
                "  bar.sync 0;\n  bar.sync 0;\n"
                "  mul.rn.f32 %f1, %f2, %f3;\n"
                "  add.rn.f32 %f4, %f1, %f5;\n"
                "  st.global.b32 [%rd3], %r1;\n"
            )
        parameters = ",\n".join(
            f"  .param .u64 .ptr .global {symbol}_param_{index}"
            for index in range(count)
        )
        bodies.append(
            f".visible .entry {symbol}(\n{parameters}\n)\n"
            f".maxntid {block[0]}, {block[1]}, {block[2]}\n"
            "{\n" + shared + "  ret;\n}\n"
        )
    ptx = ".version 7.1\n.target sm_86\n.address_size 64\n" + "".join(bodies)
    binary = (
        "module {\n  gpu.binary @module "
        "[#gpu.object<#nvvm.target<chip = \"sm_86\">, "
        f'properties = {{assembly = "{_mlir_string(ptx)}"}}, "PTX">]\n}}\n'
    )
    return MlirPtxArtifact(
        binary_mlir=binary,
        ptx=ptx,
        source_sha256=hashlib.sha256(source.encode("utf-8")).hexdigest(),
        command=("mock-mlir-opt",),
        mlir_opt_version="LLVM 24.0.0git",
        mlir_opt_source_revision="a" * 40,
        target_chip="sm_86",
        target_features="+ptx71",
        ptx_version="7.1",
        entries=entries,
    )


def _lower(source: str, **kwargs):
    return _artifact(source, tuple(kwargs["expected_entries"]))


def _compile(tile=(16, 16, 16), *, rows=2500):
    with (
        patch("gac.mlir_toolchain.lower_gpu_mlir_to_ptx", side_effect=_lower),
        patch("gac.mlir_tiled_matmul.lower_gpu_mlir_to_ptx", side_effect=_lower),
    ):
        return compile_mlir_tiled_ffn(
            _graph(rows),
            tile_m=tile[0],
            tile_n=tile[1],
            tile_k=tile[2],
            command=("mock-mlir-opt",),
        )


class MlirTiledFfnProgramTests(unittest.TestCase):
    def test_three_dispatch_plan_executes_the_recorded_tile(self):
        program = _compile()
        self.assertEqual(program.target, TARGET)
        self.assertEqual(program.matmul_tile, (16, 16, 16))
        self.assertEqual(program.thread_block, (16, 16, 1))
        self.assertEqual(len(program.calls), 3)
        self.assertEqual(
            [call.implementation for call in program.calls],
            [IMPLEMENTATION, IMPLEMENTATION, "mlir-ptx"],
        )
        self.assertEqual(len(program.calls[0].inputs), 3)
        self.assertEqual(program.calls[0].output, program.base_program.calls[1].output)
        self.assertEqual(program.calls[0].grid, (32, 157, 1))
        self.assertEqual(program.calls[1].grid, (16, 157, 1))
        description = program.describe()
        self.assertEqual(description["dispatch_count"], 3)
        self.assertEqual(description["matmul_tile"], [16, 16, 16])
        self.assertTrue(description["single_buffered_shared_memory"])
        self.assertFalse(description["async_double_buffered"])

    def test_uniad_profile_derives_rows_and_launches_from_base_plan(self):
        program = _compile(rows=40000)
        self.assertEqual(program.rows, 40000)
        self.assertEqual(program.fc1.config.m, 40000)
        self.assertEqual(program.fc2.config.m, 40000)
        self.assertEqual(program.calls[0].grid, (32, 2500, 1))
        self.assertEqual(program.calls[1].grid, (16, 2500, 1))
        self.assertEqual(program.calls[2].grid, (40000, 1, 1))
        self.assertIn("memref<40000x256xf32", program.fc1.gpu_source)
        self.assertIn("memref<40000x512xf32", program.fc2.gpu_source)
        self.assertEqual(
            program.describe()["profile"],
            {
                "query_shape": [1, 40000, 256],
                "rows": 40000,
                "channels": 256,
                "hidden_channels": 512,
            },
        )

    def test_8x8x16_is_a_distinct_compiled_candidate(self):
        small = _compile((8, 8, 16))
        default = _compile((16, 16, 16))
        self.assertEqual(small.matmul_tile, (8, 8, 16))
        self.assertEqual(small.thread_block, (8, 8, 1))
        self.assertEqual(small.fc1.config.block, small.thread_block)
        self.assertEqual(small.calls[0].block, small.thread_block)
        self.assertNotEqual(small.fc1.gpu_source, default.fc1.gpu_source)
        self.assertNotEqual(small.integrity_sha256, default.integrity_sha256)

    def test_metadata_and_artifact_tampering_fail_closed(self):
        program = _compile()
        for invalid in (True, 2.0):
            with self.subTest(abi_version=invalid), self.assertRaisesRegex(
                BackendError, "execution metadata"
            ):
                validate_mlir_tiled_ffn_program(
                    replace(program, abi_version=invalid)
                )
        with self.assertRaisesRegex(BackendError, "integrity"):
            validate_mlir_tiled_ffn_program(
                replace(program, integrity_sha256="0" * 64)
            )
        with self.assertRaisesRegex(BackendError, "same autotuned tile"):
            validate_mlir_tiled_ffn_program(
                replace(
                    program,
                    fc1=replace(
                        program.fc1,
                        config=replace(program.fc1.config, tile_k=8),
                    ),
                )
            )

    def test_weight_free_artifact_round_trip_and_tampering(self):
        program = _compile((8, 8, 16))
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "tiled"
            manifest = write_mlir_tiled_ffn_artifact(artifact, program)
            self.assertEqual(
                manifest["artifact_schema_version"], ARTIFACT_SCHEMA_VERSION
            )
            self.assertEqual(manifest["matmul_tile"], [8, 8, 16])
            with (
                patch(
                    "gac.mlir_toolchain.lower_gpu_mlir_to_ptx",
                    side_effect=AssertionError("must not compile"),
                ),
                patch(
                    "gac.mlir_tiled_matmul.lower_gpu_mlir_to_ptx",
                    side_effect=AssertionError("must not compile"),
                ),
            ):
                loaded = load_mlir_tiled_ffn_artifact(artifact, _graph())
            self.assertEqual(loaded.integrity_sha256, program.integrity_sha256)

            manifest_path = artifact / "manifest.json"
            value = json.loads(manifest_path.read_text(encoding="utf-8"))
            value["matmul_tile"] = [16, 16, 16]
            manifest_path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(BackendError, "digest"):
                load_mlir_tiled_ffn_artifact(artifact, _graph())

    def test_legacy_schema_one_bev_artifact_still_loads(self):
        program = _compile((8, 8, 16))
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "legacy-tiled"
            write_mlir_tiled_ffn_artifact(artifact, program)
            manifest_path = artifact / "manifest.json"
            value = json.loads(manifest_path.read_text(encoding="utf-8"))
            value.pop("profile")
            value["artifact_schema_version"] = LEGACY_ARTIFACT_SCHEMA_VERSION
            value.pop("manifest_payload_sha256")
            payload = json.dumps(
                value,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            value["manifest_payload_sha256"] = hashlib.sha256(payload).hexdigest()
            manifest_path.write_text(
                json.dumps(value, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

            loaded = load_mlir_tiled_ffn_artifact(artifact, _graph())
            self.assertEqual(loaded.rows, 2500)
            self.assertEqual(loaded.integrity_sha256, program.integrity_sha256)

    def test_uniad_weight_free_artifact_round_trip_preserves_profile(self):
        program = _compile((8, 8, 16), rows=40000)
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "uniad-tiled"
            manifest = write_mlir_tiled_ffn_artifact(artifact, program)
            self.assertEqual(manifest["profile"]["rows"], 40000)
            self.assertEqual(
                manifest["components"]["fc1"]["config"]["grid"],
                [64, 5000, 1],
            )
            with (
                patch(
                    "gac.mlir_toolchain.lower_gpu_mlir_to_ptx",
                    side_effect=AssertionError("must not compile"),
                ),
                patch(
                    "gac.mlir_tiled_matmul.lower_gpu_mlir_to_ptx",
                    side_effect=AssertionError("must not compile"),
                ),
            ):
                loaded = load_mlir_tiled_ffn_artifact(artifact, _graph(40000))
            self.assertEqual(loaded.rows, 40000)
            self.assertEqual(loaded.fc1.config.grid, (64, 5000, 1))
            self.assertEqual(loaded.fc2.config.grid, (32, 5000, 1))
            self.assertEqual(loaded.integrity_sha256, program.integrity_sha256)

    def test_autotune_candidate_is_derived_from_the_compiled_program(self):
        program = _compile((8, 8, 16))
        candidate = tiled_ffn_candidate(program)
        self.assertEqual(candidate.backend_strategy, TARGET)
        self.assertEqual(candidate.matmul_tile, program.matmul_tile)
        self.assertEqual(candidate.thread_block, program.thread_block)
        self.assertEqual(len(executable_ffn_candidates(program)), 3)

        class Input:
            def set(self, _value, *, stream):
                self.stream = stream

        class Session:
            def __init__(self, selected_program):
                self.program = selected_program
                self._cp = SimpleNamespace(cuda=SimpleNamespace())
                self._device = _Context()
                self._stream = _Stream()
                self._inputs = {"query": Input()}
                from threading import Lock

                self._lock = Lock()
                self._closed = False
                self._failed = False
                self.info = {
                    "device_id": 0,
                    "name": "fake-sm86",
                    "compute_capability": [8, 6],
                }

            def _execute_calls(self):
                pass

            def run(self, _inputs):
                return SimpleNamespace(
                    outputs={"output": np.zeros((1, 2500, 256), dtype=np.float32)}
                )

        graph = _graph()
        sessions = {
            CUBLAS_PTX_TARGET: Session(
                SimpleNamespace(target=CUBLAS_PTX_TARGET, graph=graph)
            ),
            MLIR_CUDA_TARGET: Session(
                SimpleNamespace(target=MLIR_CUDA_TARGET, graph=graph)
            ),
            TARGET: Session(program),
        }
        runner = FfnSessionAutotuneRunner(
            np.zeros((1, 2500, 256), dtype=np.float32), sessions
        )
        self.assertIsNone(runner.unsupported_reason(candidate))
        altered = replace(candidate, matmul_tile=(16, 16, 16))
        self.assertIn("differs", runner.unsupported_reason(altered))
        self.assertEqual(
            runner.run(candidate)["output"].shape, (1, 2500, 256)
        )


class _Context:
    def __enter__(self):
        return self

    def __exit__(self, *_):
        return None


class _Stream(_Context):
    def __init__(self):
        self.synchronizations = 0

    def synchronize(self):
        self.synchronizations += 1


class _Array:
    _pointer = 0x1000

    def __init__(self, shape, dtype, value=None):
        self.shape = tuple(shape)
        self.dtype = np.dtype(dtype)
        self.flags = SimpleNamespace(c_contiguous=True)
        self.data = SimpleNamespace(ptr=_Array._pointer)
        _Array._pointer += 0x100000
        self.value = np.empty(self.shape, dtype=self.dtype)
        if value is not None:
            self.value[...] = value

    def set(self, value, *, stream):
        self.value[...] = value


class _Kernel:
    def __init__(self, symbol):
        self.symbol = symbol
        self.attributes = {"max_threads_per_block": 1024}
        self.calls = []

    def __call__(self, grid, block, arguments, *, stream):
        self.calls.append((grid, block, arguments, stream))


class _Module:
    def __init__(self, path):
        self.path = Path(path)
        self.ptx = self.path.read_text(encoding="utf-8")
        self.kernels = {}

    def get_function(self, symbol):
        kernel = _Kernel(symbol)
        self.kernels[symbol] = kernel
        return kernel


class _Cupy:
    def __init__(self):
        self.device = _Context()
        self.stream = _Stream()
        self.modules = []
        self.cuda = SimpleNamespace(
            Device=lambda _device: self.device,
            Stream=lambda **_kwargs: self.stream,
        )

    def RawModule(self, **kwargs):
        module = _Module(kwargs["path"])
        self.modules.append(module)
        return module

    @staticmethod
    def array(value, *, dtype, order, copy):
        return _Array(np.asarray(value).shape, dtype, value)

    @staticmethod
    def empty(shape, *, dtype, order):
        return _Array(shape, dtype)


def _device_info(capability=(8, 6)):
    return {
        "device_id": 0,
        "name": "fake-sm86",
        "compute_capability": list(capability),
    }


class MlirTiledFfnRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.program = _compile()

    def test_validation_precedes_cuda_and_device_must_be_sm86(self):
        with patch("gac.mlir_tiled_ffn_runtime.cuda_device_info") as probe:
            with self.assertRaisesRegex(ValidationError, "MlirTiledFfnProgram"):
                MlirTiledFfnSession(object())
        probe.assert_not_called()

        fake = _Cupy()
        with (
            patch(
                "gac.mlir_tiled_ffn_runtime.cuda_device_info",
                return_value=_device_info((8, 9)),
            ),
            patch("gac.mlir_tiled_ffn_runtime._cupy", return_value=fake) as cupy,
        ):
            with self.assertRaisesRegex(DeviceUnavailableError, "8.6"):
                MlirTiledFfnSession(self.program)
        cupy.assert_not_called()

    def test_loads_three_ptx_modules_and_submits_exactly_three_kernels(self):
        fake = _Cupy()
        with (
            patch(
                "gac.mlir_tiled_ffn_runtime.cuda_device_info",
                return_value=_device_info(),
            ),
            patch("gac.mlir_tiled_ffn_runtime._cupy", return_value=fake),
        ):
            session = MlirTiledFfnSession(self.program)

        self.assertEqual(len(fake.modules), 3)
        self.assertEqual(set(session._kernels), {call.symbol for call in self.program.calls})
        self.assertEqual(session.info["matmul_backend"]["tile"], [16, 16, 16])
        session._execute_calls()
        calls = [
            session._kernels[call.symbol].calls[0] for call in self.program.calls
        ]
        self.assertEqual([len(item[2]) for item in calls], [4, 3, 6])
        self.assertEqual([item[0] for item in calls], [call.grid for call in self.program.calls])
        paths = list(session._ptx_paths.values())
        self.assertTrue(all(path.exists() for path in paths))
        session.close()
        self.assertTrue(all(not path.exists() for path in paths))


if __name__ == "__main__":
    unittest.main()
