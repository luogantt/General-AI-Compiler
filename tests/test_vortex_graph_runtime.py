"""Full-graph host ABI checks independent of an installed Vortex driver."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from gac.backends.vortex_graph_runtime import emit_graph_host, emit_graph_makefile, emit_graph_verifier
from gac.errors import BackendError
from gac.ir import Graph, Node, TensorSpec
from gac.memory import plan_memory


def _graph() -> Graph:
    spec = TensorSpec((8,), "float16")
    return Graph(
        name="raw half graph",
        inputs={"x": spec},
        constants={"scale": np.ones((8,), dtype=np.float16)},
        outputs=("z", "x", "scale"),
        nodes=[Node("multiply", "Mul", ("x", "scale"), ("hidden",)),
               Node("activate", "ReLU", ("hidden",), ("z",))],
        value_specs={"hidden": spec, "z": spec},
    )


def _dispatches() -> list[dict]:
    return [
        {"kernel": "multiply", "node_id": "multiply", "op": "Mul",
         "inputs": ["x", "scale"], "output": "hidden", "elements": 8},
        {"kernel": "activate", "node_id": "activate", "op": "ReLU",
         "inputs": ["hidden"], "output": "z", "elements": 8},
    ]


class VortexGraphRuntimeTests(unittest.TestCase):
    def test_plan_from_another_graph_is_rejected(self) -> None:
        graph = _graph()
        with self.assertRaisesRegex(BackendError, "different graph"):
            emit_graph_host(graph, replace(plan_memory(graph), graph_name="other"), _dispatches())

    def test_unknown_buffer_is_rejected(self) -> None:
        graph = _graph()
        dispatches = _dispatches()
        dispatches[0]["inputs"] = ["unplanned"]
        with self.assertRaisesRegex(BackendError, "unplanned"):
            emit_graph_host(graph, plan_memory(graph), dispatches)

    def test_rv32_launch_limit_is_checked(self) -> None:
        graph = _graph()
        dispatches = _dispatches()
        dispatches[0]["elements"] = 1 << 32
        with self.assertRaisesRegex(BackendError, "RV32"):
            emit_graph_host(graph, plan_memory(graph), dispatches)

    def test_generated_host_compiles_against_vortex2(self) -> None:
        source_root = os.environ.get("GAC_VORTEX_SOURCE")
        compiler = shutil.which("g++")
        if not source_root or not compiler:
            self.skipTest("set GAC_VORTEX_SOURCE and install g++ for native host ABI verification")
        include = Path(source_root) / "sw/runtime/include"
        self.assertTrue((include / "vortex2.h").is_file())
        graph = _graph()
        with tempfile.TemporaryDirectory() as directory:
            host = Path(directory) / "main.cpp"
            host.write_text(emit_graph_host(graph, plan_memory(graph), _dispatches()), encoding="utf-8")
            result = subprocess.run(
                [compiler, "-std=c++17", "-Wall", "-Wextra", "-pedantic", "-Werror",
                 "-fsyntax-only", "-I", str(include), str(host)],
                capture_output=True, text=True, timeout=60, check=False,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_makefile_keeps_full_lowering_in_one_pinned_tree(self) -> None:
        makefile = emit_graph_makefile()
        self.assertIn("--convert-scf-to-cf", makefile)
        self.assertIn("--convert-math-to-llvm", makefile)
        self.assertIn("--convert-arith-to-llvm", makefile)
        self.assertIn("--convert-cf-to-llvm", makefile)
        self.assertIn("VX_CFLAGS += -ffixed-x31", makefile)
        self.assertIn("VOLT_LLVM_REVISION_EXPECTED", makefile)
        self.assertNotIn("common.h", makefile)


class VortexGraphArtifactVerifierTests(unittest.TestCase):
    def _artifact(self, root: Path) -> dict:
        sources = {
            "01-import.json", "02-graph.json", "03-passes.json", "04-capabilities.json",
            "05-dispatch-dag.json", "06-templates.json", "07-tiling.json", "08-schedule.json",
            "10-runtime.json", "kernel.mlir", "main.cpp", "Makefile", "verify_outputs.py",
            "inputs/input_0000.bin", "constants/constant_0000.bin", "goldens/golden_0000.bin",
        }
        for name in sources:
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"\0\0" if name.endswith(".bin") else b"source\n")
        verifier = root / "verify_artifact.py"
        verifier.write_text(emit_graph_verifier(), encoding="utf-8")
        sources.add("verify_artifact.py")
        files = {name: {"size_bytes": (root / name).stat().st_size,
                        "sha256": hashlib.sha256((root / name).read_bytes()).hexdigest()}
                 for name in sources}
        record = lambda role: [{"name": "tensor", "file": f"{role}/{role[:-1]}_0000.bin",
                                "shape": [1], "dtype": "float16", "size_bytes": 2}]
        manifest = {"schema_version": 1, "compiler_backend": "vortex-whole-graph", "files": files,
                    "tensor_files": {role: record(role) for role in ("inputs", "constants", "outputs")},
                    "source_sha256": files["kernel.mlir"]["sha256"]}
        self._save_manifest(root, manifest)
        return manifest

    def _save_manifest(self, root: Path, manifest: dict) -> None:
        (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    def _verify(self, root: Path) -> subprocess.CompletedProcess:
        return subprocess.run([sys.executable, str(root / "verify_artifact.py")],
                              capture_output=True, text=True, check=False, timeout=20)

    def test_accepts_sources_and_ignores_derived_build_products(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._artifact(root)
            (root / "kernel.elf").write_bytes(b"derived")
            (root / "outputs").mkdir()
            (root / "outputs/output_0000.bin").write_bytes(b"\0\0")
            result = self._verify(root)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertTrue(json.loads(result.stdout)["artifact_verified"])

    def test_detects_tampered_input_without_loading_numpy(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._artifact(root)
            (root / "inputs/input_0000.bin").write_bytes(b"\0\1")
            result = self._verify(root)
            self.assertEqual(result.returncode, 1)
            self.assertIn("SHA-256 mismatch", result.stderr)

    def test_rejects_missing_required_source_record(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self._artifact(root)
            del manifest["files"]["main.cpp"]
            self._save_manifest(root, manifest)
            result = self._verify(root)
            self.assertEqual(result.returncode, 1)
            self.assertIn("file set mismatch", result.stderr)

    def test_rejects_extra_source_and_path_traversal_records(self) -> None:
        for name in ("another.cpp", "../main.cpp"):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                manifest = self._artifact(root)
                manifest["files"][name] = manifest["files"]["main.cpp"]
                self._save_manifest(root, manifest)
                result = self._verify(root)
                self.assertEqual(result.returncode, 1)
                self.assertIn("file set mismatch", result.stderr)

    def test_requires_all_or_none_of_input_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self._artifact(root)
            manifest["tensor_files"]["inputs"].append({"name": "other", "file": "inputs/input_0001.bin",
                                                        "shape": [1], "dtype": "float16", "size_bytes": 2})
            self._save_manifest(root, manifest)
            result = self._verify(root)
            self.assertEqual(result.returncode, 1)
            self.assertIn("incomplete input", result.stderr)


if __name__ == "__main__":
    unittest.main()
