"""Public API, executable CLI and model/target integration after layering."""

from __future__ import annotations

from contextlib import redirect_stdout
import importlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

from gac.compiler.pipeline import pipeline_manifest
from gac.core.errors import BackendError
from gac.core.ir import Graph, Node, TensorSpec


REPO = Path(__file__).resolve().parents[1]


def _graph() -> Graph:
    return Graph("compat_relu", {"x": TensorSpec((2, 3), "float32")}, ("y",),
                 [Node("relu", "ReLU", ("x",), ("y",))])


def _subprocess(*args: str) -> subprocess.CompletedProcess[str]:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(REPO / "src"), str(REPO), environment.get("PYTHONPATH", "")]
    )
    environment["CUDA_VISIBLE_DEVICES"] = ""
    return subprocess.run([sys.executable, *args], cwd=REPO, env=environment,
                          capture_output=True, text=True, timeout=30, check=False)


class RefactorIdentityTests(unittest.TestCase):
    def test_old_and_new_imports_share_modules_classes_and_exceptions_in_either_order(self):
        aliases = {
            "gac.ir": "gac.core.ir",
            "gac.errors": "gac.core.errors",
            "gac.memory": "gac.compiler.memory",
            "gac.passes": "gac.compiler.passes",
            "gac.backends.vortex_graph": "gac.targets.vortex.compiler",
            "gac.bevformer_ffn": "gac.models.bevformer.ffn",
            "gac.uniad": "gac.models.uniad",
            "gac.uniad.contract": "gac.models.uniad.contract",
            "gac.cli": "gac.tools.cli",
        }
        for canonical_first in (False, True):
            with self.subTest(canonical_first=canonical_first):
                script = f"""
import importlib
aliases = {aliases!r}
for legacy, canonical in aliases.items():
    first, second = (canonical, legacy) if {canonical_first!r} else (legacy, canonical)
    left = importlib.import_module(first)
    right = importlib.import_module(second)
    assert left is right, (legacy, canonical)
from gac.ir import Graph as OldGraph, Node as OldNode, TensorSpec as OldSpec
from gac.core.ir import Graph, Node, TensorSpec
from gac.errors import BackendError as OldError
from gac.core.errors import BackendError
assert (OldGraph, OldNode, OldSpec, OldError) == (Graph, Node, TensorSpec, BackendError)
value = OldGraph('legacy-object', {{'x': OldSpec((1,), 'float32')}}, ('y',),
                 [OldNode('bad', 'UnsupportedCompatibilityProbe', ('x',), ('y',))])
from gac.targets.vortex.compiler import compile_vortex_graph
try:
    compile_vortex_graph(value)
except BackendError as error:
    assert isinstance(error, OldError)
else:
    raise AssertionError('canonical compiler must reject unsupported legacy Graph')
"""
                result = _subprocess("-c", script)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_mocking_either_public_path_changes_the_same_compiler_global(self):
        legacy = importlib.import_module("gac.backends.vortex_graph")
        canonical = importlib.import_module("gac.targets.vortex.compiler")
        self.assertIs(legacy, canonical)
        for patch_path, entry in (
            ("gac.backends.vortex_graph.verify_mlir", canonical.compile_vortex_graph),
            ("gac.targets.vortex.compiler.verify_mlir", legacy.compile_vortex_graph),
        ):
            with self.subTest(patch_path=patch_path), mock.patch(patch_path, return_value="verified-lowered") as verify:
                self.assertIs(legacy.verify_mlir, canonical.verify_mlir)
                program = entry(_graph(), mlir_opt="not-an-installed-tool")
                self.assertEqual(program.lowered_mlir, "verified-lowered")
                self.assertEqual(verify.call_args.kwargs["command"][0], "not-an-installed-tool")
                self.assertEqual(verify.call_count, 1)


class RefactorCliTests(unittest.TestCase):
    def test_python_m_legacy_and_canonical_help_have_valid_runpy_origin(self):
        for module in ("gac.cli", "gac.tools.cli"):
            with self.subTest(module=module):
                result = _subprocess("-m", module, "--help")
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn("usage:", result.stdout)
                self.assertIn("compile", result.stdout)
                self.assertNotIn("Traceback", result.stderr)

    def test_python_m_legacy_and_canonical_compile_identical_source_without_a_gpu(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model.json"
            model.write_text(_graph().to_json(), encoding="utf-8")
            sources = []
            for index, module in enumerate(("gac.cli", "gac.tools.cli")):
                output = root / f"compiled-{index}.cu"
                with self.subTest(module=module):
                    result = _subprocess("-m", module, "compile", str(model), "-o", str(output))
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    report = json.loads(result.stdout)
                    self.assertEqual(report["artifact"], "cuda-source")
                    self.assertEqual(report["source_file"], str(output))
                    self.assertTrue(output.with_suffix(".plan.json").is_file())
                    sources.append(output.read_text(encoding="utf-8"))
            self.assertEqual(sources[0], sources[1])
            self.assertIn("__global__", sources[0])

    def test_bevformer_weights_option_selects_registered_adapter_and_preserves_hash(self):
        from gac.tools import build_vortex_graph

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "trusted.gacweights"
            output = root / "artifact"
            with mock.patch.object(build_vortex_graph, "load_model_graph", return_value=_graph()) as load, redirect_stdout(io.StringIO()):
                code = build_vortex_graph.main([
                    "--bevformer-weights", str(source), "--expected-checkpoint-sha256", "a" * 64,
                    "--output", str(output),
                ])
            self.assertEqual(code, 0)
            load.assert_called_once_with("bevformer-step", source, expected_checkpoint_sha256="a" * 64)
            self.assertTrue((output / "kernel.mlir").is_file())

    def test_uniad_bundle_compiles_to_ten_vortex_artifacts_with_cuda_imports_blocked(self):
        from gac.models.uniad.artifact import write_bundle
        from gac.models.uniad.contract import UniADFFNContract

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            contract = UniADFFNContract(layer=2, batch=1, bev_h=1, bev_w=2, embed_dim=4, hidden_dim=8)
            rng = np.random.default_rng(124)
            weights = {name: (np.ones(shape, np.float32) if name == "norm_weight" else
                              rng.normal(0.0, 0.1, shape).astype(np.float32))
                       for name, shape in contract.weight_shapes.items()}
            bundle = root / "tiny.uniadffn"
            write_bundle(bundle, contract, weights)
            output = root / "artifact"
            # A fresh process proves no optional GPU import is hidden by this
            # test suite's previously cached modules.
            script = r"""
from importlib.abc import MetaPathFinder
import sys
blocked = ('cupy', 'torch', 'gac.targets.cuda', 'gac.execution.cuda',
           'gac.backends.cuda', 'gac.models.bevformer')
class NoGpuImports(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == name or fullname.startswith(name + '.') for name in blocked):
            raise AssertionError('Vortex UniAD path imported an unrelated GPU/model module: ' + fullname)
sys.meta_path.insert(0, NoGpuImports())
from gac.tools.build_vortex_graph import main
code = main(sys.argv[1:])
assert not [name for name in sys.modules if any(name == prefix or name.startswith(prefix + '.') for prefix in blocked)]
raise SystemExit(code)
"""
            result = _subprocess("-c", script, "--model", str(bundle), "--adapter", "uniad-ffn",
                                 "--output", str(output))
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            stdout = json.loads(result.stdout)
            self.assertEqual(stdout["target"], "vortex-v3")
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["stages"], pipeline_manifest())
            self.assertEqual(manifest["fallback_targets"], [])
            self.assertGreater(manifest["dispatch_count"], 0)
            for stage in manifest["stages"]:
                self.assertTrue((output / stage["artifact"]).is_file(), stage["name"])
            graph_record = json.loads((output / "02-graph.json").read_text(encoding="utf-8"))
            metadata = graph_record["metadata"]
            self.assertEqual(metadata["model_family"], "uniad")
            self.assertEqual(metadata["model_adapter"], "uniad-ffn")
            self.assertEqual(metadata["model_scope"], "stage2-bev-encoder-single-layer-ffn")
            self.assertFalse(metadata["full_model"])
            self.assertEqual(metadata["layer"], contract.layer)
            self.assertTrue(all(source.startswith("UniAD.") for node in graph_record["nodes"]
                                for source in node["source_nodes"]))
            verify = _subprocess(str(output / "verify_artifact.py"))
            self.assertEqual(verify.returncode, 0, verify.stdout + verify.stderr)


if __name__ == "__main__":
    unittest.main()
