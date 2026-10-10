"""Contracts for the explicit MLIR CUDA/cuBLAS execution slice."""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
import io
import json
import os
import subprocess
from threading import Lock
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from gac.backends.mlir_cuda import (
    CUDA_GRID_X_MAX,
    CUDA_KERNEL_INDEX_MAX,
    MlirCudaProgram,
    compile_mlir_cuda,
    validate_mlir_cuda_program,
)
from gac.bevformer_ffn import (
    BEVFormerFFNConfig,
    build_bevformer_ffn_graph,
    ffn_weight_shapes,
)
from gac.cli import main
from gac.errors import BackendError, ValidationError
from gac.ir import TensorSpec
from gac.mlir_runtime import MlirCudaSession
from gac.mlir_toolchain import MlirOptNotFoundError, discover_mlir_opt
from gac.runtime import reference_run


def _ffn_fixture(
    *,
    bev_h: int = 1,
    bev_w: int = 2,
    embed_dim: int = 4,
) -> tuple[MlirCudaProgram, np.ndarray]:
    config = BEVFormerFFNConfig(
        bev_h=bev_h,
        bev_w=bev_w,
        embed_dim=embed_dim,
    )
    rng = np.random.default_rng(20261010)
    weights = {
        name: rng.normal(0.0, 0.1, shape).astype(np.float32)
        for name, shape in ffn_weight_shapes(config).items()
    }
    query = rng.normal(
        0.0,
        0.1,
        (config.batch, config.queries, config.embed_dim),
    ).astype(np.float32)
    graph = build_bevformer_ffn_graph(weights, config, name="mlir_cuda_ffn")
    return compile_mlir_cuda(graph), query


def _invoke(arguments: list[str]) -> tuple[int, str, str]:
    stdout = io.StringIO()
    stderr = io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        code = main(arguments)
    return code, stdout.getvalue(), stderr.getvalue()


class MlirCudaCompilerTests(unittest.TestCase):
    def test_ffn_has_two_explicit_cublas_calls_and_two_fused_kernels(self):
        program, _ = _ffn_fixture()

        self.assertEqual(
            [call.symbol for call in program.calls],
            [
                "gac_cublas_fc1_matmul_f32",
                "gac_bias_relu_inplace_f32",
                "gac_cublas_fc2_matmul_f32",
                "gac_bias_residual_layer_norm_f32",
            ],
        )
        self.assertEqual(
            [call.implementation for call in program.calls],
            ["cupy-cublas", "nvrtc-cuda", "cupy-cublas", "nvrtc-cuda"],
        )
        self.assertEqual(
            [call.kind for call in program.calls],
            ["matmul", "fused-kernel", "matmul", "fused-kernel"],
        )
        self.assertEqual(program.calls[1].aliases_input, 0)
        self.assertEqual(program.calls[1].grid, (1, 1, 1))
        self.assertEqual(program.calls[1].block, (256, 1, 1))
        self.assertEqual(program.calls[3].grid, (2, 1, 1))
        self.assertEqual(program.calls[3].block, (256, 1, 1))
        self.assertEqual(
            program.calls[3].node_ids,
            ("fc2_bias_add", "residual_add", "post_layer_norm"),
        )
        self.assertTrue(program.calls[3].source_nodes)

    def test_runtime_call_mlir_is_reviewable_and_call_metadata_is_one_to_one(self):
        program, _ = _ffn_fixture()

        for call in program.calls:
            self.assertEqual(program.source.count(f"func.call @{call.symbol}"), 1)
            self.assertEqual(program.source.count(f"private @{call.symbol}"), 1)
        self.assertEqual(program.source.count("func.call @"), len(program.calls))
        self.assertIn("memref.alloc", program.source)
        self.assertIn("memref.dealloc", program.source)
        self.assertNotIn("linalg.matmul", program.source)
        self.assertIn("linalg.matmul", program.tensor_source)

        description = program.describe()
        self.assertEqual(description["compiler_backend"], "mlir-cuda-sm86")
        self.assertEqual(description["target"], "mlir-cuda-sm86")
        self.assertEqual(description["target_intent"], "cuda-sm86")
        self.assertEqual(description["compute_capability"], [8, 6])
        self.assertEqual(description["lowering_stage"], "runtime-call-plan")
        self.assertTrue(description["executable"])
        self.assertEqual(description["matmul_lowering"], "cupy-cublas")
        self.assertEqual(description["calls"], [call.to_dict() for call in program.calls])
        self.assertFalse(description["mlir_source_executed"])
        self.assertEqual(description["runtime_dispatch_source"], "validated-call-records")
        self.assertEqual(len(description["integrity"]["integrity_sha256"]), 64)
        self.assertEqual(program.calls[0].abi_version, 2)
        self.assertEqual(
            (program.calls[0].m, program.calls[0].n, program.calls[0].k),
            (2, 8, 4),
        )
        self.assertEqual(program.calls[0].dtype, "float32")
        self.assertEqual(program.calls[0].layout, "contiguous")
        self.assertEqual(program.calls[3].epsilon, np.float32(1.0e-5))

    def test_layer_norm_kernel_uses_one_block_reduction_and_is_fused_with_residual(self):
        program, _ = _ffn_fixture()
        source = program.kernel_source

        self.assertIn("gac_bias_residual_layer_norm_f32", source)
        self.assertIn("__shared__ float warp_sums[8]", source)
        self.assertIn("__shfl_down_sync", source)
        self.assertIn("rsqrtf", source)
        self.assertIn("projected[index] + bias[column] + residual[index]", source)
        self.assertEqual(source.count('extern "C" __global__ void'), 2)
        self.assertNotIn("gac_cublas_fc1_matmul_f32", source)
        self.assertNotIn("gac_cublas_fc2_matmul_f32", source)

    def test_near_match_without_post_layer_norm_is_rejected_without_fallback(self):
        config = BEVFormerFFNConfig(
            bev_h=1,
            bev_w=2,
            embed_dim=4,
            post_layer_norm=False,
        )
        rng = np.random.default_rng(7)
        weights = {
            name: rng.normal(size=shape).astype(np.float32)
            for name, shape in ffn_weight_shapes(config).items()
        }
        graph = build_bevformer_ffn_graph(weights, config)

        with self.assertRaisesRegex(BackendError, "requires the fused BEVFormer FFN"):
            compile_mlir_cuda(graph)

    def test_fused_layer_norm_rejects_a_non_last_axis(self):
        program, _ = _ffn_fixture()
        graph = program.reference_graph.clone()
        graph.nodes = [
            replace(
                node,
                attrs={**node.attrs, "axis": 1},
            )
            if node.op == "LayerNorm"
            else node
            for node in graph.nodes
        ]

        with self.assertRaisesRegex(BackendError, r"last (?:normalization )?axis"):
            compile_mlir_cuda(graph)

    def test_cuda_float_literal_is_valid_for_integral_epsilon(self):
        program, _ = _ffn_fixture()
        graph = program.reference_graph.clone()
        graph.nodes = [
            replace(
                node,
                attrs={**node.attrs, "epsilon": 1.0},
            )
            if node.op == "LayerNorm"
            else node
            for node in graph.nodes
        ]

        integral_epsilon = compile_mlir_cuda(graph)
        self.assertNotRegex(integral_epsilon.kernel_source, r"\+\s+1f\s*\)")
        self.assertRegex(
            integral_epsilon.kernel_source,
            r"\+\s+1(?:\.0+|(?:\.\d+)?e[+-]?0+)f\s*\)",
        )

    def test_generated_kernels_reject_shapes_outside_signed_index_domain(self):
        program, _ = _ffn_fixture()
        query = next(iter(program.reference_graph.inputs))
        channels = int(program.reference_graph.inputs[query].shape[-1])
        hidden_channels = int(
            program.reference_graph.constants[program.calls[0].inputs[1]].shape[-1]
        )

        safe_rows = CUDA_KERNEL_INDEX_MAX // hidden_channels
        safe_graph = program.reference_graph.clone()
        safe_spec = TensorSpec((safe_rows, channels), "float32")
        safe_graph.inputs[query] = safe_spec
        safe_graph.value_specs[query] = safe_spec
        safe_program = compile_mlir_cuda(safe_graph)
        hidden_elements = safe_rows * hidden_channels
        self.assertEqual(
            safe_program.calls[1].grid,
            ((hidden_elements + 255) // 256, 1, 1),
        )

        oversized_graph = program.reference_graph.clone()
        oversized_spec = TensorSpec((safe_rows + 1, channels), "float32")
        oversized_graph.inputs[query] = oversized_spec
        oversized_graph.value_specs[query] = oversized_spec
        with self.assertRaisesRegex(BackendError, "signed 32-bit kernel index"):
            compile_mlir_cuda(oversized_graph)

    def test_runtime_manifest_escapes_node_ids_and_avoids_symbol_collisions(self):
        program, _ = _ffn_fixture()
        graph = program.reference_graph.clone()
        graph.name = "gac_cublas_fc1_matmul_f32"
        unusual_id = 'fc1 "quoted" \\ path\nnext'
        graph.nodes[0] = replace(graph.nodes[0], id=unusual_id)

        unusual = compile_mlir_cuda(graph)
        self.assertNotIn(f'gac.node_id = "{unusual_id}"', unusual.source)
        self.assertNotIn(
            "func.func @gac_cublas_fc1_matmul_f32(",
            unusual.source,
        )

    def test_upstream_mlir_opt_accepts_explicit_runtime_call_module(self):
        required = os.environ.get("GAC_REQUIRE_MLIR") == "1"
        if not os.environ.get("GAC_MLIR_OPT") and not required:
            self.skipTest("set GAC_MLIR_OPT to run the MLIR parser/verifier")
        try:
            executable = str(discover_mlir_opt())
        except MlirOptNotFoundError as exc:
            if required:
                self.fail(f"GAC_REQUIRE_MLIR=1 but mlir-opt is unavailable: {exc}")
            self.skipTest(str(exc))
        program, _ = _ffn_fixture()
        unusual_graph = program.reference_graph.clone()
        unusual_graph.name = "gac_cublas_fc1_matmul_f32"
        unusual_graph.nodes[0] = replace(
            unusual_graph.nodes[0],
            id='fc1 "quoted" \\ path\nnext',
        )
        for candidate in (program, compile_mlir_cuda(unusual_graph)):
            result = subprocess.run(
                [executable, "--verify-each"],
                input=candidate.source,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)


class MlirCudaRuntimeContractTests(unittest.TestCase):
    def test_non_program_is_rejected_before_cuda_is_touched(self):
        with self.assertRaisesRegex(ValidationError, "MlirCudaProgram"):
            MlirCudaSession(object())

    def test_close_releases_owned_module_and_stream_references(self):
        class FakeDevice:
            def __enter__(self):
                return self

            def __exit__(self, *_):
                return None

        class FakeStream:
            def __init__(self):
                self.synchronize_calls = 0

            def synchronize(self):
                self.synchronize_calls += 1

        stream = FakeStream()
        module = object()
        session = MlirCudaSession.__new__(MlirCudaSession)
        session._lock = Lock()
        session._closed = False
        session._device = FakeDevice()
        session._stream = stream
        session._module = module
        session._values = {"value": object()}
        session._workspaces = {"workspace": object()}
        session._outputs = {"output": object()}
        session._inputs = {"input": object()}
        session._constants = {"constant": object()}
        session._kernels = {"kernel": object()}

        session.close()

        self.assertEqual(stream.synchronize_calls, 1)
        self.assertTrue(session._closed)
        self.assertIsNone(session._module)
        self.assertIsNone(session._stream)
        self.assertFalse(session._values)
        self.assertFalse(session._kernels)

    def test_initialization_error_releases_module_and_stream_references(self):
        program, _ = _ffn_fixture()

        class FakeContext:
            def __enter__(self):
                return self

            def __exit__(self, *_):
                return None

        class FakeStream(FakeContext):
            def __init__(self):
                self.synchronize_calls = 0

            def synchronize(self):
                self.synchronize_calls += 1

        class FakeModule:
            def compile(self):
                return None

            def get_function(self, _):
                return object()

        device = FakeContext()
        stream = FakeStream()
        module = FakeModule()
        fake_cuda = SimpleNamespace(
            Device=lambda _: device,
            Stream=lambda **_: stream,
        )
        fake_cupy = SimpleNamespace(
            cuda=fake_cuda,
            RawModule=lambda **_: module,
            array=lambda *_, **__: object(),
            empty=lambda *_, **__: object(),
        )
        session = MlirCudaSession.__new__(MlirCudaSession)

        with (
            patch(
                "gac.mlir_runtime.cuda_device_info",
                return_value={"compute_capability": [8, 6]},
            ),
            patch("gac.mlir_runtime._cupy", return_value=fake_cupy),
            patch.object(MlirCudaSession, "_allocate_call_values"),
            patch.object(MlirCudaSession, "_validate_call_plan"),
            patch.object(
                MlirCudaSession,
                "_call_info",
                side_effect=RuntimeError("attribute query failed"),
            ),
        ):
            with self.assertRaisesRegex(
                BackendError, "session initialization failed"
            ):
                session.__init__(program)

        self.assertEqual(stream.synchronize_calls, 2)
        self.assertTrue(session._closed)
        self.assertIsNone(session._module)
        self.assertIsNone(session._stream)
        self.assertFalse(session._values)
        self.assertFalse(session._kernels)

    def test_initialization_interrupts_release_resources_and_propagate_unchanged(self):
        program, _ = _ffn_fixture()

        class FakeContext:
            def __enter__(self):
                return self

            def __exit__(self, *_):
                return None

        class FakeStream(FakeContext):
            def __init__(self):
                self.synchronize_calls = 0

            def synchronize(self):
                self.synchronize_calls += 1

        class FakeModule:
            def compile(self):
                return None

            def get_function(self, _):
                return object()

        for failure in (KeyboardInterrupt("cancelled"), SystemExit(17)):
            with self.subTest(failure=type(failure).__name__):
                device = FakeContext()
                stream = FakeStream()
                module = FakeModule()
                fake_cuda = SimpleNamespace(
                    Device=lambda _: device,
                    Stream=lambda **_: stream,
                )
                fake_cupy = SimpleNamespace(
                    cuda=fake_cuda,
                    RawModule=lambda **_: module,
                    array=lambda *_, **__: object(),
                    empty=lambda *_, **__: object(),
                )
                session = MlirCudaSession.__new__(MlirCudaSession)

                def interrupt_after_allocation(_):
                    self.assertIs(session._module, module)
                    self.assertIs(session._stream, stream)
                    self.assertTrue(session._values)
                    self.assertTrue(session._workspaces)
                    self.assertTrue(session._outputs)
                    self.assertTrue(session._inputs)
                    self.assertTrue(session._constants)
                    self.assertTrue(session._kernels)
                    raise failure

                with (
                    patch(
                        "gac.mlir_runtime.cuda_device_info",
                        return_value={"compute_capability": [8, 6]},
                    ),
                    patch("gac.mlir_runtime._cupy", return_value=fake_cupy),
                    patch.object(MlirCudaSession, "_validate_call_plan"),
                    patch.object(
                        MlirCudaSession,
                        "_call_info",
                        side_effect=interrupt_after_allocation,
                    ),
                ):
                    with self.assertRaises(type(failure)) as raised:
                        session.__init__(program)

                self.assertIs(raised.exception, failure)
                self.assertEqual(stream.synchronize_calls, 2)
                self.assertTrue(session._closed)
                self.assertIsNone(session._module)
                self.assertIsNone(session._stream)
                self.assertFalse(session._values)
                self.assertFalse(session._workspaces)
                self.assertFalse(session._outputs)
                self.assertFalse(session._inputs)
                self.assertFalse(session._constants)
                self.assertFalse(session._kernels)

    def test_dispatch_is_driven_by_explicit_call_records(self):
        program, _ = _ffn_fixture()
        calls: list[tuple[object, object, object]] = []
        launches: list[tuple[object, object, object, object]] = []

        class FakeCupy:
            @staticmethod
            def matmul(left, right, *, out):
                calls.append((left, right, out))

        class FakeKernel:
            def __call__(self, grid, block, arguments, *, stream):
                launches.append((grid, block, arguments, stream))

        session = MlirCudaSession.__new__(MlirCudaSession)
        session._cp = FakeCupy()
        session._stream = object()
        names = {
            name
            for call in program.calls
            for name in (*call.inputs, call.output)
        }
        session._values = {name: object() for name in names}
        # The Add+ReLU result deliberately aliases the first MatMul workspace.
        session._values[program.calls[1].output] = session._values[
            program.calls[1].inputs[0]
        ]
        session._kernels = {
            call.symbol: FakeKernel()
            for call in program.calls
            if call.implementation == "nvrtc-cuda"
        }

        for call in program.calls:
            session._execute_call(call)

        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0][0], session._values["query"])
        self.assertEqual(calls[0][1], session._values["fc1_weight"])
        self.assertEqual(calls[0][2], session._values["fc1_mm"])
        self.assertEqual(calls[1][0], session._values["activated"])
        self.assertEqual(calls[1][1], session._values["fc2_weight"])
        self.assertEqual(calls[1][2], session._values["fc2_mm"])
        self.assertEqual(len(launches), 2)
        self.assertEqual(
            launches[0][2],
            (session._values["fc1_mm"], session._values["fc1_bias"]),
        )
        self.assertEqual(
            launches[1][2],
            tuple(session._values[name] for name in program.calls[3].inputs)
            + (session._values["output"],),
        )
        self.assertIs(launches[0][3], session._stream)
        self.assertIs(launches[1][3], session._stream)

    def test_unknown_runtime_symbol_is_rejected_instead_of_falling_back(self):
        program, _ = _ffn_fixture()
        invalid_call = replace(program.calls[0], symbol="unknown_runtime_symbol")
        invalid_program = replace(program, calls=(invalid_call, *program.calls[1:]))
        session = MlirCudaSession.__new__(MlirCudaSession)
        session.program = invalid_program
        session._values = {
            name: object()
            for call in invalid_program.calls
            for name in (*call.inputs, call.output)
        }
        session._kernels = {
            call.symbol: object()
            for call in invalid_program.calls
            if call.implementation == "nvrtc-cuda"
        }

        with self.assertRaisesRegex(BackendError, "MLIR runtime"):
            session._validate_call_plan()

    def test_artifact_tampering_is_rejected_before_cuda_discovery(self):
        program, _ = _ffn_fixture()
        validate_mlir_cuda_program(program)
        altered_call = replace(program.calls[0], alpha=0.5)
        cases = (
            replace(program, source=program.source + "// altered\n"),
            replace(program, kernel_source=program.kernel_source + "// altered\n"),
            replace(program, calls=(altered_call, *program.calls[1:])),
            replace(program, integrity_sha256="0" * 64),
        )
        for altered in cases:
            with self.subTest(component=altered):
                with patch("gac.mlir_runtime.cuda_device_info") as device_probe:
                    with self.assertRaisesRegex(BackendError, "artifact"):
                        MlirCudaSession(altered)
                device_probe.assert_not_called()

    def test_runtime_validates_declared_alias_identity_and_tensor_abi(self):
        program, _ = _ffn_fixture()
        values: dict[str, np.ndarray] = {
            name: np.empty(spec.shape, dtype=spec.dtype)
            for name, spec in program.graph.value_specs.items()
            if name in {*program.graph.inputs, *program.graph.constants}
        }
        for call in program.calls:
            if call.aliases_input is not None:
                values[call.output] = values[call.inputs[call.aliases_input]]
            else:
                spec = program.graph.value_specs[call.output]
                values[call.output] = np.empty(spec.shape, dtype=spec.dtype)

        session = MlirCudaSession.__new__(MlirCudaSession)
        session.program = program
        session._graph = program.graph
        session._values = values
        session._validate_actual_values()

        aliased = program.calls[1]
        session._values[aliased.output] = np.empty_like(
            session._values[aliased.inputs[0]]
        )
        with self.assertRaisesRegex(BackendError, "alias identity"):
            session._validate_actual_values()

        session._values[aliased.output] = session._values[aliased.inputs[0]]
        session._values[program.calls[3].output] = np.empty(
            (1,), dtype=np.float32
        )
        with self.assertRaisesRegex(BackendError, "requires"):
            session._validate_actual_values()

    def test_runtime_abi_rejects_oversized_index_and_grid_domains(self):
        program, _ = _ffn_fixture()
        call = program.calls[1]
        assert call.channels is not None

        graph = program.graph.clone()
        rows = CUDA_KERNEL_INDEX_MAX // call.channels + 1
        elements = rows * call.channels
        oversized_spec = TensorSpec((rows, call.channels), "float32")
        graph.value_specs[call.inputs[0]] = oversized_spec
        graph.value_specs[call.output] = oversized_spec
        oversized_call = replace(
            call,
            rows=rows,
            elements=elements,
            grid=((elements + 255) // 256, 1, 1),
        )
        session = MlirCudaSession.__new__(MlirCudaSession)
        session.program = SimpleNamespace(graph=graph)

        with self.assertRaisesRegex(BackendError, "32-bit kernel index"):
            session._validate_call_abi(oversized_call)

        session.program = SimpleNamespace(graph=program.graph)
        with self.assertRaisesRegex(BackendError, "grid.x limit"):
            session._validate_call_abi(
                replace(call, grid=(CUDA_GRID_X_MAX + 1, 1, 1))
            )

    @unittest.skipUnless(
        os.environ.get("GAC_REQUIRE_CUDA") == "1",
        "set GAC_REQUIRE_CUDA=1 to run the physical SM86 integration",
    )
    def test_physical_sm86_output_matches_reference(self):
        program, query = _ffn_fixture(bev_h=2, bev_w=3, embed_dim=32)
        expected = reference_run(program.reference_graph, {"query": query})

        with MlirCudaSession(program) as session:
            result = session.run({"query": query})

        np.testing.assert_allclose(
            result.outputs["output"],
            expected["output"],
            rtol=1.0e-4,
            atol=1.0e-5,
        )
        self.assertGreater(result.gpu_stream_ms, 0.0)
        self.assertGreater(result.wall_ms, 0.0)

    @unittest.skipUnless(
        os.environ.get("GAC_REQUIRE_CUDA") == "1",
        "set GAC_REQUIRE_CUDA=1 to run the physical SM86 integration",
    )
    def test_physical_sm86_accepts_integral_layer_norm_epsilon(self):
        base, query = _ffn_fixture(bev_h=1, bev_w=2, embed_dim=32)
        graph = base.reference_graph.clone()
        graph.nodes = [
            replace(node, attrs={**node.attrs, "epsilon": 1.0})
            if node.op == "LayerNorm"
            else node
            for node in graph.nodes
        ]
        program = compile_mlir_cuda(graph)
        expected = reference_run(program.reference_graph, {"query": query})

        with MlirCudaSession(program) as session:
            result = session.run({"query": query})

        np.testing.assert_allclose(
            result.outputs["output"],
            expected["output"],
            rtol=1.0e-4,
            atol=1.0e-5,
        )


class MlirCudaCliTests(unittest.TestCase):
    def test_run_selects_explicit_mlir_cuda_program_and_reports_calls(self):
        program, _ = _ffn_fixture()

        class FakeSession:
            def __init__(self, selected, *, device_id):
                self.program = selected
                self.device_id = device_id
                self.info = {
                    "name": "fake SM86",
                    "compute_capability": [8, 6],
                    "compiler_backend": "mlir-cuda-sm86",
                }
                self.setup_ms = 3.0
                self.kernel_info = [call.to_dict() for call in selected.calls]

            def __enter__(self):
                return self

            def __exit__(self, *_):
                return None

            def run(self, inputs):
                return SimpleNamespace(
                    outputs=reference_run(self.program.reference_graph, inputs),
                    wall_ms=1.25,
                    gpu_stream_ms=0.25,
                )

        with (
            patch("gac.cli.load_graph", return_value=program.reference_graph),
            patch("gac.cli.compile_mlir_cuda", return_value=program) as compile_mock,
            patch("gac.cli.MlirCudaSession", FakeSession),
            patch("gac.cli.compile_cuda") as legacy_compile,
        ):
            code, stdout, stderr = _invoke(
                [
                    "run",
                    "ffn.json",
                    "--backend",
                    "mlir-cuda-sm86",
                    "--device",
                    "2",
                    "--warmup",
                    "1",
                    "--repeat",
                    "2",
                    "--verify",
                ]
            )

        self.assertEqual(code, 0, stderr)
        self.assertEqual(stderr, "")
        compile_mock.assert_called_once_with(program.reference_graph)
        legacy_compile.assert_not_called()
        report = json.loads(stdout)
        self.assertEqual(report["backend"], "mlir-cuda-sm86")
        self.assertEqual(report["device"]["compute_capability"], [8, 6])
        self.assertEqual(report["verification"], "passed")
        self.assertEqual(report["repeat"], 2)
        self.assertEqual(report["wall_ms"]["median"], 1.25)
        self.assertEqual(report["gpu_stream_ms"]["median"], 0.25)
        self.assertEqual(len(report["runtime_calls"]), 4)
        self.assertEqual(report["lowering"]["matmul_lowering"], "cupy-cublas")

    def test_cli_rejects_options_that_would_hide_or_replace_the_call_plan(self):
        program, _ = _ffn_fixture()
        cases = (
            (["--no-fuse"], "requires its explicit FFN fusion plan"),
            (
                ["--conv-backend", "generated"],
                "--conv-backend does not apply",
            ),
            (
                ["--cudnn-tensor-core", "always"],
                "--cudnn-tensor-core does not apply",
            ),
            (
                ["--matmul-backend", "generated"],
                "explicit cuBLAS MatMul lowering",
            ),
        )
        for extra, diagnostic in cases:
            with self.subTest(extra=extra):
                with patch(
                    "gac.cli.load_graph", return_value=program.reference_graph
                ):
                    code, stdout, stderr = _invoke(
                        [
                            "run",
                            "ffn.json",
                            "--backend",
                            "mlir-cuda-sm86",
                            *extra,
                        ]
                    )
                self.assertEqual(code, 1)
                self.assertEqual(stdout, "")
                self.assertIn(diagnostic, stderr)


if __name__ == "__main__":
    unittest.main()
