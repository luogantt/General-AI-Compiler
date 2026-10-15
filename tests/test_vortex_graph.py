from __future__ import annotations

import os
from pathlib import Path
import tempfile
import subprocess
import sys
import unittest

import numpy as np

from gac.backends.vortex_graph import compile_vortex_graph, write_vortex_graph_artifact
from gac.errors import BackendError
from gac.ir import Graph, Node, TensorSpec
from gac.ops import SUPPORTED_OPS
from gac.runtime import reference_run


def chain_graph():
    return Graph("vortex_chain", {"x": TensorSpec((2, 3), "float32")}, ("y",), [
        Node("plus", "Add", ("x", "bias"), ("a",)),
        Node("relu", "ReLU", ("a",), ("b",)),
        Node("twice", "Add", ("b", "b"), ("c",)),
        Node("negate", "Sub", ("c", "bias"), ("y",)),
    ], constants={"bias": np.asarray([0.1, -0.2, 0.3], np.float32)})


class VortexGraphTests(unittest.TestCase):
    def test_single_device_plan_and_live_buffer_reuse(self):
        graph = chain_graph()
        original_specs = dict(graph.value_specs)
        program = compile_vortex_graph(graph)
        self.assertEqual(graph.value_specs, original_specs)  # does not mutate its caller
        self.assertEqual([s["stage"] for s in program.stages], list(range(1, 11)))
        self.assertEqual({d["target"] for d in program.dispatches}, {"vortex-v3"})
        self.assertEqual(program.describe()["fallback_targets"], [])
        plan = program.memory_plan
        self.assertEqual(plan.allocation("a").offset, plan.allocation("c").offset)
        self.assertNotEqual(plan.allocation("a").offset, plan.allocation("b").offset)
        self.assertEqual([d["wait_for_dispatch"] for d in program.dispatches], [None, 0, 1, 2])
        self.assertIn("scf.if", program.source)
        self.assertNotIn("cuda", program.source.lower())
        self.assertIn("llvm.riscv.vx.uniform", program.source)
        self.assertIn("@gac_op_0000", program.source)

    def test_rejects_unknown_ops_dynamic_and_nonpositive_shapes(self):
        with self.assertRaisesRegex(BackendError, "UnsupportedThing"):
            compile_vortex_graph(Graph("bad", {"x": TensorSpec((2,), "float32")}, ("y",), [Node("bad", "UnsupportedThing", ("x",), ("y",))]))
        for shape in (("batch", 3), (0, 3)):
            with self.assertRaisesRegex(BackendError, "positive static shape"):
                compile_vortex_graph(Graph("bad", {"x": TensorSpec(shape, "float32")}, ("y",), [Node("copy", "Identity", ("x",), ("y",))]))

    def test_artifact_records_all_ten_stages_and_named_tensor_abi(self):
        program = compile_vortex_graph(chain_graph())
        values = {"x": np.arange(6, dtype=np.float32).reshape(2, 3)}
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "artifact"
            report = write_vortex_graph_artifact(output, program, inputs=values, golden=reference_run(program.graph, values))
            for stage in program.stages:
                self.assertTrue((output / stage["artifact"]).is_file())
            self.assertTrue((output / "inputs/input_0000.bin").is_file())
            self.assertTrue((output / "constants/constant_0000.bin").is_file())
            self.assertTrue((output / "goldens/golden_0000.bin").is_file())
            self.assertEqual(report["supported_software_ops"], sorted(SUPPORTED_OPS))
            self.assertFalse((output / "kernel.cpp").exists())
            with self.assertRaisesRegex(BackendError, "not empty"):
                write_vortex_graph_artifact(output, program)

    def test_whole_bevformer_step_compiles_every_node_without_cuda(self):
        from tests.test_bevformer_step_pipeline import small_config, make_weights
        from gac.bevformer_step_pipeline import build_bevformer_step_pipeline_graph
        config = small_config()
        encoder, decoder = make_weights(config)
        graph = build_bevformer_step_pipeline_graph(encoder, decoder, config)
        program = compile_vortex_graph(graph)
        self.assertGreater(len(program.dispatches), 500)
        self.assertTrue({"Conv2D", "GridSample2D", "MSDeformableAttention", "Softmax", "LayerNorm"}.issubset({d["op"] for d in program.dispatches}))
        self.assertEqual({d["target"] for d in program.dispatches}, {"vortex-v3"})

    def test_scalar_output_verifier_accepts_correct_and_rejects_wrong_data(self):
        graph = Graph("scalar", {"x": TensorSpec((3,), "float32")}, ("y",),
                      [Node("sum", "ReduceSum", ("x",), ("y",), {"axes": [0], "keepdims": False})])
        program = compile_vortex_graph(graph)
        inputs = {"x": np.asarray([1, 2, 3], np.float32)}
        golden = {"y": np.asarray(6, np.float32)}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_vortex_graph_artifact(root, program, inputs=inputs, golden=golden)
            (root / "outputs").mkdir()
            for value, expected_code in ((6, 0), (7, 1)):
                np.asarray(value, np.float32).tofile(root / "outputs/output_0000.bin")
                result = subprocess.run([sys.executable, str(root / "verify_outputs.py"), "--data-dir", str(root)],
                                        capture_output=True, text=True, timeout=20)
                self.assertEqual(result.returncode, expected_code, result.stdout + result.stderr)


@unittest.skipUnless(os.environ.get("GAC_VORTEX_LLVM_BIN"), "set GAC_VORTEX_LLVM_BIN for real MLIR/native compilation")
class VortexGraphNativeTests(unittest.TestCase):
    def test_native_chain(self):
        from gac.backends.vortex_graph_validation import NativeVortexGraphSession
        program = compile_vortex_graph(chain_graph())
        values = {"x": np.asarray([[-1, 2, -3], [4, -5, 6]], np.float32)}
        with NativeVortexGraphSession(program, llvm_bin=os.environ["GAC_VORTEX_LLVM_BIN"]) as session:
            actual = session.run(values)
        expected = reference_run(program.graph, values)
        np.testing.assert_allclose(actual["y"], expected["y"], rtol=1e-6, atol=1e-6)

    def test_whole_bevformer_cold_and_warm_frames(self):
        from gac.backends.vortex_graph_validation import NativeVortexGraphSession
        from tests.test_bevformer_step_pipeline import small_config, make_weights, runtime_inputs
        from gac.bevformer_step_pipeline import build_bevformer_step_pipeline_graph
        config = small_config()
        encoder, decoder = make_weights(config)
        graph = build_bevformer_step_pipeline_graph(encoder, decoder, config)
        tools = os.environ["GAC_VORTEX_LLVM_BIN"]
        program = compile_vortex_graph(graph, mlir_opt=str(Path(tools) / "mlir-opt"))
        with NativeVortexGraphSession(program, llvm_bin=tools) as session:
            cold = runtime_inputs(config, warm=False)
            warm = runtime_inputs(config, warm=True)
            for label, values in (("cold", cold), ("warm", warm)):
                expected = reference_run(program.graph, values)
                actual = session.run(values)
                for name in program.graph.outputs:
                    with self.subTest(frame=label, output=name):
                        np.testing.assert_allclose(actual[name], expected[name], rtol=2e-3, atol=3e-4)


if __name__ == "__main__":
    unittest.main()
