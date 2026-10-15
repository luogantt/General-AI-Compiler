from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

from gac.backends.cuda import compile_cuda
from gac.bevformer_step_pipeline import build_bevformer_step_pipeline_graph
from gac.ir import Graph, Node, TensorSpec
from gac.package import load_package, write_package
from gac.passes import ShapeInferencePass
from gac.tools.plan_bevformer import build_plan_report, main
from tests.test_bevformer_step_pipeline import make_weights, small_config


def _write_test_hxp(path: Path, *, complete_step: bool = True) -> None:
    graph = Graph(
        name="tiny_complete_bevformer_step",
        inputs={"images": TensorSpec((1, 4), "float32")},
        outputs=("raw_detections",),
        nodes=[
            Node(
                "images_to_float32",
                "Identity",
                ("images",),
                ("raw_detections",),
            )
        ],
        metadata={
            "kind": (
                "bevformer-complete-stateful-streaming-step-static"
                if complete_step
                else "bevformer-ffn-static"
            )
        },
    )
    ShapeInferencePass().run(graph)
    write_package(path, compile_cuda(graph))


class BevFormerPlanCliTests(unittest.TestCase):
    def test_fused_full_step_hxp_still_discovers_all_nine_ffn_regions(self):
        config = small_config()
        encoder_weights, decoder_weights = make_weights(config)
        graph = build_bevformer_step_pipeline_graph(
            encoder_weights,
            decoder_weights,
            config,
            name="complete_step_hxp_plan_test",
        )
        self.assertEqual(len(graph.nodes), 1053)

        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory) / "step.hxp"
            write_package(package, compile_cuda(graph))
            packaged_graph = load_package(package).graph
            self.assertLess(len(packaged_graph.nodes), len(graph.nodes))

            report = build_plan_report(package)
            regions = report["capability_analysis"]["vortex_ffn_regions"]
            self.assertEqual(len(regions), 9)
            self.assertTrue(all(region["fused_bias_relu"] for region in regions))
            self.assertEqual(report["dispatches_by_target"]["vortex-v3"], 9)
            self.assertFalse(report["execution_status"]["physical_runtime_ready"])

    def test_writes_deterministic_cuda_only_plan_and_reviewable_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = root / "step.hxp"
            output = root / "plan.json"
            source_dir = root / "sources"
            _write_test_hxp(package)
            arguments = [
                str(package),
                "--output",
                str(output),
                "--cuda-only",
                "--alignment",
                "64",
                "--no-verify-cuda-fallback",
                "--lowered-source-dir",
                str(source_dir),
            ]

            with redirect_stdout(io.StringIO()):
                self.assertEqual(main(arguments), 0)
            first = output.read_bytes()
            with redirect_stdout(io.StringIO()):
                self.assertEqual(main(arguments), 0)
            self.assertEqual(output.read_bytes(), first)

            report = json.loads(first)
            self.assertEqual(report["kind"], "bevformer-whole-model-heterogeneous-offline-plan")
            self.assertEqual(report["dispatches_by_target"], {"cuda-sm86": 1})
            self.assertFalse(report["cuda_fallback_verified"])
            self.assertEqual(report["runtime_plan"]["global_memory_plan"]["alignment"], 64)
            self.assertTrue(report["input_package"]["integrity_verified"])
            self.assertFalse(report["command_execution"]["physical_runtime_executed"])
            self.assertEqual(report["source_emission"]["artifact_count"], 1)
            source_file = report["source_emission"]["artifacts"][0]["source_file"]
            self.assertEqual(Path(source_file).suffix, ".cu")
            self.assertTrue((source_dir / source_file).is_file())
            self.assertFalse(list(root.glob(".plan.json.*.tmp")))

    def test_rejects_non_full_step_hxp(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = root / "ffn.hxp"
            output = root / "plan.json"
            _write_test_hxp(package, complete_step=False)
            errors = io.StringIO()
            with redirect_stdout(io.StringIO()), redirect_stderr(errors):
                self.assertEqual(
                    main([str(package), "--output", str(output)]),
                    1,
                )
            self.assertIn("complete stateful BEVFormer streaming step", errors.getvalue())
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
