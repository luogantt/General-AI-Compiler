"""Focused contracts for the BEVFormer FFN -> MLIR -> Vortex backend."""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

from gac.backends.vortex import compile_vortex, write_vortex_artifact
from gac.bevformer_ffn import (
    BEVFormerFFNConfig,
    build_bevformer_ffn_graph,
    ffn_weight_shapes,
)
from gac.errors import BackendError
from gac.ir import Graph
from gac.runtime import reference_run
from gac.tools import build_bevformer_vortex
from gac.tools.build_bevformer_vortex import _synthetic_graph


_MLIR_OPT = "/opt/llvm/bin/mlir-opt"
_MLIR_VERSION = "LLVM version 24.0.0git"
_MLIR_REVISION = "7" * 40


def _ffn_graph(
    *,
    post_layer_norm: bool = True,
    name: str = "bevformer_vortex_test",
) -> Graph:
    config = BEVFormerFFNConfig(
        bev_h=1,
        bev_w=2,
        embed_dim=4,
        post_layer_norm=post_layer_norm,
    )
    rng = np.random.default_rng(3070)
    weights = {
        weight_name: np.ascontiguousarray(
            rng.normal(0.0, 0.125, shape), dtype=np.float32
        )
        for weight_name, shape in ffn_weight_shapes(config).items()
    }
    return build_bevformer_ffn_graph(weights, config, name=name)


def _compile_with_mock_mlir(graph: Graph):
    def emulate_mlir_opt_printer(source: str, **_kwargs) -> str:
        if "@bevformer_vortex_plan_" in source:
            return re.sub(
                r"\bfunc\.call(?=\s+@gac_vortex_)", "call", source
            )
        return source

    with (
        mock.patch(
            "gac.backends.vortex.verify_mlir",
            side_effect=emulate_mlir_opt_printer,
        ) as verify,
        mock.patch(
            "gac.backends.vortex.probe_mlir_opt_version",
            return_value=_MLIR_VERSION,
        ) as probe_version,
        mock.patch(
            "gac.backends.vortex.probe_mlir_opt_source_revision",
            return_value=_MLIR_REVISION,
        ) as probe_revision,
    ):
        program = compile_vortex(
            graph,
            mlir_opt=_MLIR_OPT,
            use_wsl=True,
            wsl_distribution="Ubuntu-24.04",
            timeout=17,
        )
    return program, verify, probe_version, probe_revision


class VortexCompileTests(unittest.TestCase):
    def test_compile_verifies_both_mlir_layers_and_binds_generated_sources(self):
        graph = _ffn_graph()
        before = graph.to_json()

        program, verify, probe_version, probe_revision = _compile_with_mock_mlir(graph)

        self.assertEqual(graph.to_json(), before)
        self.assertEqual(program.plan.rows, 2)
        self.assertEqual(program.plan.channels, 4)
        self.assertEqual(program.plan.hidden, 8)
        self.assertEqual(program.plan.input_shape, (1, 2, 4))
        self.assertEqual(
            program.plan.kernels,
            (
                "bevformer_fc1_bias_relu",
                "bevformer_fc2_bias_residual",
                "bevformer_layer_norm",
            ),
        )

        self.assertEqual(verify.call_count, 2)
        high_level_call, call_plan_call = verify.call_args_list
        self.assertEqual(high_level_call.args, (program.mlir_program.source,))
        self.assertEqual(program.optimized_mlir, high_level_call.args[0])
        expected_toolchain_arguments = {
            "mlir_opt": _MLIR_OPT,
            "use_wsl": True,
            "wsl_distribution": "Ubuntu-24.04",
            "timeout": 17,
        }
        self.assertEqual(high_level_call.kwargs, expected_toolchain_arguments)
        self.assertEqual(call_plan_call.kwargs, expected_toolchain_arguments)
        self.assertEqual(
            program.vortex_mlir,
            re.sub(
                r"\bfunc\.call(?=\s+@gac_vortex_)",
                "call",
                call_plan_call.args[0],
            ),
        )

        optimized_hash = hashlib.sha256(
            program.optimized_mlir.encode("utf-8")
        ).hexdigest()
        self.assertIn(
            f"@bevformer_vortex_plan_{optimized_hash}", program.vortex_mlir
        )
        private_symbols = re.findall(
            r"func\.func\s+private\s+@(gac_vortex_[A-Za-z0-9_]+)",
            program.vortex_mlir,
        )
        called_symbols = re.findall(
            r"(?:func\.)?call\s+@(gac_vortex_[A-Za-z0-9_]+)",
            program.vortex_mlir,
        )
        self.assertEqual(len(private_symbols), 3)
        self.assertEqual(called_symbols, private_symbols)
        self.assertEqual(len(set(private_symbols)), 3)
        self.assertIn("fc1", private_symbols[0])
        self.assertIn("fc2", private_symbols[1])
        self.assertIn("layer_norm", private_symbols[2])
        self.assertRegex(program.vortex_mlir, r"memref<1x2x4xf32>")
        self.assertRegex(program.vortex_mlir, r"memref<4x8xf32>")
        self.assertNotRegex(program.vortex_mlir, r"memref<[^>]*\?")
        epsilon_offset = program.vortex_mlir.index("%epsilon = arith.constant")
        norm_call = re.search(
            rf"(?:func\.)?call\s+@{re.escape(private_symbols[2])}",
            program.vortex_mlir,
        )
        self.assertIsNotNone(norm_call)
        assert norm_call is not None
        norm_call_offset = norm_call.start()
        self.assertLess(epsilon_offset, norm_call_offset)

        probe_version.assert_called_once_with(
            mlir_opt=_MLIR_OPT,
            use_wsl=True,
            wsl_distribution="Ubuntu-24.04",
            timeout=17,
        )
        probe_revision.assert_called_once_with(
            _MLIR_OPT,
            use_wsl=True,
            wsl_distribution="Ubuntu-24.04",
            timeout=17,
        )

        self.assertIn(optimized_hash, program.kernel_source)
        self.assertIn(optimized_hash, program.model_config_header)
        for symbol in program.plan.kernels:
            self.assertIn(f"__kernel void {symbol}", program.kernel_source)
            self.assertIn(f'vx_module_get_kernel(module, "{symbol}"', program.host_source)
        self.assertIn("#include <vx_spawn2.h>", program.kernel_source)
        self.assertIn("#include <vortex2.h>", program.host_source)
        self.assertIn("vx_enqueue_launch", program.host_source)
        self.assertIn("uint32_t parse_u32_option", program.host_source)
        self.assertIn("requires decimal digits only", program.host_source)
        self.assertIn("exceeds uint32 range", program.host_source)
        self.assertIn(
            'warmup = parse_u32_option(argv[++i], "--warmup")',
            program.host_source,
        )
        self.assertIn(
            'repeat = parse_u32_option(argv[++i], "--repeat")',
            program.host_source,
        )
        self.assertNotIn("std::stoul", program.host_source)

        description = program.describe()
        self.assertEqual(description["compiler_backend"], "vortex-v3")
        self.assertEqual(description["dispatch_count"], 3)
        self.assertEqual(description["lowering"]["post_opt_mlir_sha256"], optimized_hash)
        self.assertEqual(
            description["lowering"]["vortex_call_plan_mlir_sha256"],
            hashlib.sha256(program.vortex_mlir.encode("utf-8")).hexdigest(),
        )
        self.assertEqual(description["lowering"]["mlir_opt_version"], _MLIR_VERSION)
        self.assertEqual(
            description["lowering"]["mlir_opt_source_revision"], _MLIR_REVISION
        )
        self.assertEqual(
            description["fallbacks"],
            {"cuda": False, "ptx": False, "cublas": False, "cupy": False},
        )

    def test_runtime_call_plan_is_parsed_for_shape_epsilon_and_kernel_order(self):
        graph = _ffn_graph()

        def change_rows(source: str) -> str:
            replacements = {
                "memref<1x2x4xf32>": "memref<1x3x4xf32>",
                "memref<1x2x8xf32>": "memref<1x3x8xf32>",
                "%fc1_launch = arith.constant 16 : index": (
                    "%fc1_launch = arith.constant 24 : index"
                ),
                "%fc2_launch = arith.constant 8 : index": (
                    "%fc2_launch = arith.constant 12 : index"
                ),
                "%norm_launch = arith.constant 2 : index": (
                    "%norm_launch = arith.constant 3 : index"
                ),
            }
            for old, new in replacements.items():
                self.assertIn(old, source)
                source = source.replace(old, new)
            return source

        def change_epsilon(source: str) -> str:
            changed, substitutions = re.subn(
                r"arith\.constant\s+[^\s:]+\s*:\s*f32",
                "arith.constant 0.125 : f32",
                source,
                count=1,
            )
            self.assertEqual(substitutions, 1)
            return changed

        def reorder_kernels(source: str) -> str:
            lines = source.splitlines(keepends=True)
            call_indices = [
                index
                for index, line in enumerate(lines)
                if re.search(r"(?:func\.)?call\s+@gac_vortex_", line)
            ]
            self.assertEqual(len(call_indices), 3)
            first, second = call_indices[:2]
            lines[first], lines[second] = lines[second], lines[first]
            return "".join(lines)

        def change_source_hash(source: str) -> str:
            changed, substitutions = re.subn(
                r"@bevformer_vortex_plan_[0-9a-f]{64}",
                "@bevformer_vortex_plan_" + "0" * 64,
                source,
                count=1,
            )
            self.assertEqual(substitutions, 1)
            return changed

        def change_weight_hash(source: str) -> str:
            changed, substitutions = re.subn(
                r'(gac\.fc1_weight_sha256\s*=\s*")[0-9a-f]{64}("\s*)',
                rf"\g<1>{'0' * 64}\g<2>",
                source,
                count=1,
            )
            self.assertEqual(substitutions, 1)
            return changed

        for name, mutate in (
            ("rows", change_rows),
            ("epsilon", change_epsilon),
            ("kernel_order", reorder_kernels),
            ("source_hash", change_source_hash),
            ("weight_hash", change_weight_hash),
        ):
            sources: list[str] = []

            def verify_two_layers(source: str, **_kwargs) -> str:
                sources.append(source)
                if len(sources) == 1:
                    return source
                pretty = re.sub(
                    r"\bfunc\.call(?=\s+@gac_vortex_)", "call", source
                )
                return mutate(pretty)

            with self.subTest(field=name):
                with (
                    mock.patch(
                        "gac.backends.vortex.verify_mlir",
                        side_effect=verify_two_layers,
                    ) as verify,
                    mock.patch("gac.backends.vortex.probe_mlir_opt_version"),
                    mock.patch(
                        "gac.backends.vortex.probe_mlir_opt_source_revision"
                    ),
                ):
                    with self.assertRaises(BackendError):
                        compile_vortex(graph, mlir_opt=_MLIR_OPT)
                self.assertEqual(verify.call_count, 2)

    def test_rejects_dense_weight_value_changed_by_high_level_mlir_opt(self):
        graph = _ffn_graph()
        verification_count = 0

        def tamper_fc1_weight(source: str, **_kwargs) -> str:
            nonlocal verification_count
            verification_count += 1
            if verification_count != 1:
                return source
            dense = re.search(
                r"dense<(?P<payload>[^>]*)>\s*:\s*tensor<4x8xf32>",
                source,
                flags=re.DOTALL,
            )
            self.assertIsNotNone(dense)
            assert dense is not None
            payload = dense.group("payload")
            changed_payload, substitutions = re.subn(
                r"[+-]?(?:(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?)",
                "9.25",
                payload,
                count=1,
            )
            self.assertEqual(substitutions, 1)
            start, end = dense.span("payload")
            return source[:start] + changed_payload + source[end:]

        with (
            mock.patch(
                "gac.backends.vortex.verify_mlir",
                side_effect=tamper_fc1_weight,
            ) as verify,
            mock.patch("gac.backends.vortex.probe_mlir_opt_version"),
            mock.patch("gac.backends.vortex.probe_mlir_opt_source_revision"),
        ):
            with self.assertRaisesRegex(
                BackendError, "dense weights.*bit-for-bit"
            ):
                compile_vortex(graph, mlir_opt=_MLIR_OPT)

        self.assertGreaterEqual(verify.call_count, 1)

    def test_accepts_mlir24_raw_hex_dense_weight_payload(self):
        graph = _ffn_graph()
        fc1_weight_bytes = np.ascontiguousarray(
            graph.constants["fc1_weight"], dtype="<f4"
        ).tobytes(order="C")
        raw_literal = f'"0x{fc1_weight_bytes.hex().upper()}"'
        verification_count = 0

        def print_fc1_weight_as_raw_hex(source: str, **_kwargs) -> str:
            nonlocal verification_count
            verification_count += 1
            if verification_count == 1:
                dense = re.search(
                    r"dense<(?P<payload>[^>]*)>\s*:\s*tensor<4x8xf32>",
                    source,
                    flags=re.DOTALL,
                )
                self.assertIsNotNone(dense)
                assert dense is not None
                start, end = dense.span("payload")
                return source[:start] + raw_literal + source[end:]
            return re.sub(
                r"\bfunc\.call(?=\s+@gac_vortex_)", "call", source
            )

        with (
            mock.patch(
                "gac.backends.vortex.verify_mlir",
                side_effect=print_fc1_weight_as_raw_hex,
            ) as verify,
            mock.patch(
                "gac.backends.vortex.probe_mlir_opt_version",
                return_value=_MLIR_VERSION,
            ),
            mock.patch(
                "gac.backends.vortex.probe_mlir_opt_source_revision",
                return_value=_MLIR_REVISION,
            ),
        ):
            program = compile_vortex(graph, mlir_opt=_MLIR_OPT)

        self.assertEqual(verify.call_count, 2)
        self.assertEqual(verification_count, 2)
        self.assertIn(f"dense<{raw_literal}> : tensor<4x8xf32>", program.optimized_mlir)
        self.assertEqual(program.plan.rows, 2)
        self.assertEqual(program.plan.channels, 4)
        self.assertEqual(program.plan.hidden, 8)
        self.assertRegex(
            program.vortex_mlir,
            r"\bcall\s+@gac_vortex_fc2_bias_residual",
        )

    def test_rejects_layer_norm_weight_and_bias_ssa_role_swap(self):
        graph = _ffn_graph()
        verification_count = 0

        def swap_normalize_parameter_uses(source: str, **_kwargs) -> str:
            nonlocal verification_count
            verification_count += 1
            if verification_count != 1:
                return source

            normalize_location = source.index('loc("post_layer_norm.normalize")')
            inputs_marker = source.rfind("} ins(", 0, normalize_location)
            self.assertNotEqual(inputs_marker, -1)
            values_start = inputs_marker + len("} ins(")
            values_end = source.index(" :", values_start, normalize_location)
            arguments = [
                value.strip()
                for value in source[values_start:values_end].split(",")
            ]
            self.assertEqual(len(arguments), 4)
            norm_weight_ssa, norm_bias_ssa = arguments[2:]
            self.assertNotEqual(norm_weight_ssa, norm_bias_ssa)
            for ssa in (norm_weight_ssa, norm_bias_ssa):
                self.assertRegex(
                    source,
                    rf"{re.escape(ssa)}\s*=\s*arith\.constant\s+dense<",
                )

            arguments[2], arguments[3] = norm_bias_ssa, norm_weight_ssa
            return (
                source[:values_start]
                + ", ".join(arguments)
                + source[values_end:]
            )

        with (
            mock.patch(
                "gac.backends.vortex.verify_mlir",
                side_effect=swap_normalize_parameter_uses,
            ) as verify,
            mock.patch("gac.backends.vortex.probe_mlir_opt_version"),
            mock.patch("gac.backends.vortex.probe_mlir_opt_source_revision"),
        ):
            with self.assertRaisesRegex(
                BackendError, "dense weight role 'norm_weight' changed"
            ):
                compile_vortex(graph, mlir_opt=_MLIR_OPT)

        self.assertEqual(verify.call_count, 1)

    def test_rejects_same_type_fc2_ssa_argument_substitution(self):
        graph = _ffn_graph()
        verification_count = 0

        def substitute_fc2_query(source: str, **_kwargs) -> str:
            nonlocal verification_count
            verification_count += 1
            if verification_count == 1:
                return source
            pretty = re.sub(
                r"\bfunc\.call(?=\s+@gac_vortex_)", "call", source
            )
            changed, substitutions = re.subn(
                r"((?:func\.)?call\s+@gac_vortex_fc2_bias_residual\s*"
                r"\(\s*%fc2_launch\s*,\s*)%query(\s*,\s*%activation\s*,)",
                r"\g<1>%residual\g<2>",
                pretty,
                count=1,
            )
            self.assertEqual(substitutions, 1)
            return changed

        with (
            mock.patch(
                "gac.backends.vortex.verify_mlir",
                side_effect=substitute_fc2_query,
            ) as verify,
            mock.patch("gac.backends.vortex.probe_mlir_opt_version"),
            mock.patch("gac.backends.vortex.probe_mlir_opt_source_revision"),
        ):
            with self.assertRaisesRegex(
                BackendError, "call 1 buffer dataflow changed"
            ):
                compile_vortex(graph, mlir_opt=_MLIR_OPT)

        self.assertEqual(verify.call_count, 2)

    def test_rejects_optimized_mlir_outside_the_audited_pattern(self):
        graph = _ffn_graph()

        def remove_one_matmul(source: str, **_kwargs) -> str:
            return source.replace("linalg.matmul", "linalg.generic", 1)

        with (
            mock.patch(
                "gac.backends.vortex.verify_mlir", side_effect=remove_one_matmul
            ) as verify,
            mock.patch(
                "gac.backends.vortex.probe_mlir_opt_version"
            ) as probe_version,
        ):
            with self.assertRaisesRegex(
                BackendError, "two matrix multiplications"
            ):
                compile_vortex(graph, mlir_opt=_MLIR_OPT)

        verify.assert_called_once()
        probe_version.assert_not_called()

    def test_rejects_non_bevformer_or_noncanonical_ffn_graphs_before_mlir(self):
        wrong_kind = _ffn_graph(name="wrong_kind")
        wrong_kind.metadata["kind"] = "generic-mlp"

        missing_norm = _ffn_graph(post_layer_norm=False, name="missing_norm")

        reordered_residual = _ffn_graph(name="reordered_residual")
        residual_index = next(
            index
            for index, node in enumerate(reordered_residual.nodes)
            if node.id == "residual_add"
        )
        residual = reordered_residual.nodes[residual_index]
        reordered_residual.nodes[residual_index] = replace(
            residual, inputs=tuple(reversed(residual.inputs))
        )

        cases = (
            (wrong_kind, "requires a canonical BEVFormer FFN graph"),
            (missing_norm, "requires constants"),
            (reordered_residual, "does not match the fused BEVFormer FFN"),
        )
        for graph, message in cases:
            with self.subTest(graph=graph.name):
                with mock.patch("gac.backends.vortex.verify_mlir") as verify:
                    with self.assertRaisesRegex(BackendError, message):
                        compile_vortex(graph, mlir_opt=_MLIR_OPT)
                verify.assert_not_called()


class VortexArtifactTests(unittest.TestCase):
    def test_artifact_contains_hash_checked_sources_and_little_endian_weights(self):
        program, *_ = _compile_with_mock_mlir(_ffn_graph())

        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "bevformer_vortex"
            report = write_vortex_artifact(destination, program)
            manifest_path = destination / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

            expected_files = {
                "Makefile",
                "common.h",
                "frontend.mlir",
                "optimized.mlir",
                "vortex.mlir",
                "kernel.cpp",
                "main.cpp",
                "model_config.h",
                "verify_artifact.py",
                "fc1_weight.bin",
                "fc1_bias.bin",
                "fc2_weight.bin",
                "fc2_bias.bin",
                "norm_weight.bin",
                "norm_bias.bin",
                "query.bin",
                "golden.bin",
            }
            self.assertEqual(set(manifest["files"]), expected_files)
            for filename, record in manifest["files"].items():
                payload = (destination / filename).read_bytes()
                self.assertEqual(record["bytes"], len(payload))
                self.assertEqual(
                    record["sha256"], hashlib.sha256(payload).hexdigest()
                )

            self.assertEqual(
                (destination / "frontend.mlir").read_text(encoding="utf-8"),
                program.mlir_program.source,
            )
            self.assertEqual(
                (destination / "optimized.mlir").read_text(encoding="utf-8"),
                program.optimized_mlir,
            )
            self.assertEqual(
                (destination / "vortex.mlir").read_text(encoding="utf-8"),
                program.vortex_mlir,
            )
            self.assertEqual(
                manifest["lowering"]["post_opt_mlir_sha256"],
                hashlib.sha256(program.optimized_mlir.encode("utf-8")).hexdigest(),
            )
            self.assertEqual(
                manifest["lowering"]["vortex_call_plan_mlir_sha256"],
                hashlib.sha256(program.vortex_mlir.encode("utf-8")).hexdigest(),
            )
            self.assertEqual(
                report["manifest_sha256"],
                hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            )
            self.assertEqual(
                Path(report["artifact_directory"]), destination.resolve()
            )

            fc1_weight = np.fromfile(destination / "fc1_weight.bin", dtype="<f4")
            np.testing.assert_array_equal(
                fc1_weight.reshape(program.graph.constants["fc1_weight"].shape),
                program.graph.constants["fc1_weight"],
            )
            query = np.fromfile(destination / "query.bin", dtype="<f4").reshape(
                program.plan.input_shape
            )
            golden = np.fromfile(destination / "golden.bin", dtype="<f4").reshape(
                program.plan.output_shape
            )
            np.testing.assert_array_equal(
                golden,
                reference_run(program.reference_graph, {"query": query})["output"],
            )

            verifier = destination / "verify_artifact.py"
            completed = subprocess.run(
                [sys.executable, str(verifier)],
                cwd=destination,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertTrue(json.loads(completed.stdout)["artifact_verified"])
            (destination / "optimized.mlir").write_text(
                "module {}\n", encoding="utf-8"
            )
            rejected = subprocess.run(
                [sys.executable, str(verifier)],
                cwd=destination,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(rejected.returncode, 1)
            self.assertIn("mismatch", rejected.stderr)

            (destination / "optimized.mlir").write_text(
                program.optimized_mlir, encoding="utf-8"
            )
            manifest["files"].pop("kernel.cpp")
            manifest_path.write_text(
                json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            missing_record = subprocess.run(
                [sys.executable, str(verifier)],
                cwd=destination,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(missing_record.returncode, 1)
            self.assertIn("file set mismatch", missing_record.stderr)
            self.assertIn("kernel.cpp", missing_record.stderr)

    def test_makefile_gates_parallel_build_and_run_targets_on_verification(self):
        program, *_ = _compile_with_mock_mlir(_ffn_graph())
        logical_makefile = re.sub(r"\\\r?\n\s*", " ", program.makefile)
        gated_targets: set[str] = set()
        for line in logical_makefile.splitlines():
            if line.startswith(("\t", "#")) or ":" not in line:
                continue
            targets, prerequisites = line.split(":", 1)
            _normal, separator, order_only = prerequisites.partition("|")
            if separator and "verify-artifact" in order_only.split():
                gated_targets.update(targets.split())

        required_build_targets = {"$(APP)", "kernel.elf", "kernel.vxbin"}
        required_run_targets = {
            "run-simx",
            "run-rtlsim",
            "run-firesim",
            "run-opae",
            "run-xrt",
            "run-aved",
        }
        self.assertTrue(
            required_build_targets <= gated_targets,
            f"parallel build targets missing verify-artifact gate: "
            f"{sorted(required_build_targets - gated_targets)}",
        )
        self.assertTrue(
            required_run_targets <= gated_targets,
            f"run targets missing verify-artifact gate: "
            f"{sorted(required_run_targets - gated_targets)}",
        )
        self.assertRegex(
            logical_makefile,
            r"(?m)^verify-artifact:\s*$\n\tpython3 verify_artifact\.py$",
        )

    def test_artifact_refuses_to_replace_nonempty_destination(self):
        program, *_ = _compile_with_mock_mlir(_ffn_graph())
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "occupied"
            destination.mkdir()
            marker = destination / "keep.txt"
            marker.write_text("keep", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "must be empty"):
                write_vortex_artifact(destination, program)

            self.assertEqual(marker.read_text(encoding="utf-8"), "keep")


class VortexBuildToolTests(unittest.TestCase):
    def test_synthetic_smoke_profile_is_deterministic_and_reviewable(self):
        first = _synthetic_graph("smoke", 41)
        second = _synthetic_graph("smoke", 41)

        self.assertEqual(first.to_json(), second.to_json())
        self.assertEqual(first.inputs["query"].shape, (1, 2, 4))
        self.assertEqual(first.constants["fc1_weight"].shape, (4, 8))
        self.assertEqual(
            first.metadata["weights_provenance"],
            {
                "kind": "deterministic-synthetic",
                "seed": 41,
                "suitable_for_accuracy_evaluation": False,
            },
        )

    def test_cli_forwards_mlir_toolchain_and_writes_external_report(self):
        program = object()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "artifact"
            report_path = root / "report.json"
            stdout = io.StringIO()
            with (
                mock.patch.object(
                    build_bevformer_vortex,
                    "compile_vortex",
                    return_value=program,
                ) as compile_backend,
                mock.patch.object(
                    build_bevformer_vortex,
                    "write_vortex_artifact",
                    return_value={"status": "generated"},
                ) as write_artifact,
                redirect_stdout(stdout),
            ):
                result = build_bevformer_vortex.main(
                    [
                        "--synthetic-width-smoke",
                        "--output",
                        str(output),
                        "--mlir-opt",
                        _MLIR_OPT,
                        "--wsl",
                        "--wsl-distribution",
                        "Ubuntu-24.04",
                        "--timeout",
                        "9",
                        "--seed",
                        "19",
                        "--report",
                        str(report_path),
                    ]
                )

            self.assertEqual(result, 0)
            graph = compile_backend.call_args.args[0]
            self.assertEqual(graph.inputs["query"].shape, (1, 1, 256))
            self.assertEqual(graph.constants["fc1_weight"].shape, (256, 512))
            self.assertEqual(
                compile_backend.call_args.kwargs,
                {
                    "mlir_opt": _MLIR_OPT,
                    "use_wsl": True,
                    "wsl_distribution": "Ubuntu-24.04",
                    "timeout": 9.0,
                },
            )
            write_artifact.assert_called_once_with(str(output), program)
            expected = {
                "status": "generated",
                "weights_provenance": {
                    "kind": "deterministic-synthetic",
                    "seed": 19,
                    "suitable_for_accuracy_evaluation": False,
                },
            }
            self.assertEqual(json.loads(stdout.getvalue()), expected)
            self.assertEqual(
                json.loads(report_path.read_text(encoding="utf-8")), expected
            )

    def test_cli_rejects_report_inside_artifact_before_compilation(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "artifact"
            stderr = io.StringIO()
            with (
                mock.patch.object(build_bevformer_vortex, "compile_vortex") as compile_backend,
                redirect_stderr(stderr),
            ):
                result = build_bevformer_vortex.main(
                    [
                        "--synthetic-smoke",
                        "--output",
                        str(output),
                        "--mlir-opt",
                        _MLIR_OPT,
                        "--report",
                        str(output / "report.json"),
                    ]
                )
            self.assertEqual(result, 1)
            self.assertIn("outside", stderr.getvalue())
            compile_backend.assert_not_called()


if __name__ == "__main__":
    unittest.main()
