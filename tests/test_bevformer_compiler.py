from __future__ import annotations

from collections import Counter
import json
import tempfile
from pathlib import Path
import unittest

import numpy as np

from gac.backends.mlir import compile_mlir
from gac.bevformer_compiler import (
    build_bevformer_whole_model_plan,
    lower_bevformer_dispatch_sources,
    plan_bevformer_graph,
)
from gac.bevformer_partition import BEVFORMER_VORTEX_TARGET
from gac.bevformer_ffn import BEVFormerFFNConfig, build_bevformer_ffn_graph
from gac.ir import Graph, Node, TensorSpec
from gac.passes import ShapeInferencePass
from tests.test_bevformer_step_pipeline import (
    make_weights as make_step_weights,
    small_config as small_step_config,
)


def _mixed_graph() -> Graph:
    config = BEVFormerFFNConfig(bev_h=1, bev_w=2, embed_dim=4)
    rng = np.random.default_rng(831)
    weights = {
        "fc1_weight": rng.normal(size=(4, 8)).astype(np.float32),
        "fc1_bias": rng.normal(size=(8,)).astype(np.float32),
        "fc2_weight": rng.normal(size=(8, 4)).astype(np.float32),
        "fc2_bias": rng.normal(size=(4,)).astype(np.float32),
        "norm_weight": rng.normal(size=(4,)).astype(np.float32),
        "norm_bias": rng.normal(size=(4,)).astype(np.float32),
    }
    ffn = build_bevformer_ffn_graph(weights, config)
    graph = Graph(
        name="mixed_bevformer_fragment",
        inputs={"images": TensorSpec((1, 2, 4), "float32")},
        outputs=("detections",),
        nodes=[
            Node("backbone.prepare", "Identity", ("images",), ("query",)),
            *ffn.nodes,
            Node("decoder.detection", "Identity", ("output",), ("detections",)),
        ],
        constants={name: value.copy() for name, value in ffn.constants.items()},
    )
    ShapeInferencePass().run(graph)
    graph.validate()
    return graph


class BevFormerWholeModelCompilerTests(unittest.TestCase):
    def test_builds_semantic_dispatch_dag_and_global_runtime_plan(self):
        plan = plan_bevformer_graph(_mixed_graph(), alignment=64)

        self.assertEqual(len(plan.capability_report.regions), 1)
        self.assertEqual(
            [
                (
                    item.target,
                    item.group.split(":", 1)[0],
                    len(item.node_ids),
                )
                for item in plan.dispatch_dag.dispatches
            ],
            [
                ("cuda-sm86", "cnn", 1),
                ("vortex-v3", "ffn", 7),
                ("cuda-sm86", "utility", 1),
            ],
        )
        self.assertEqual(plan.dispatch_dag.schedule, tuple(
            dispatch.id for dispatch in plan.dispatch_dag.dispatches
        ))
        self.assertEqual(
            [strategy.kind for strategy in plan.strategies],
            ["cnn", "ffn", "utility"],
        )
        self.assertEqual(plan.strategies[1].schedule, "dxa-double-buffer")
        self.assertFalse(plan.physical_runtime_ready)
        manifest = plan.to_dict()
        self.assertEqual(manifest["node_count"], 9)
        self.assertEqual(manifest["dispatches_by_target"]["vortex-v3"], 1)
        self.assertIn("global_memory_plan", manifest["runtime_plan"])

    def test_cuda_only_keeps_semantic_boundaries(self):
        plan = plan_bevformer_graph(
            _mixed_graph(), enable_vortex_ffn=False
        )
        self.assertEqual(
            {dispatch.target for dispatch in plan.dispatch_dag.dispatches},
            {"cuda-sm86"},
        )
        self.assertEqual(
            [dispatch.group.split(":", 1)[0] for dispatch in plan.dispatch_dag.dispatches],
            ["cnn", "ffn", "utility"],
        )
        self.assertEqual(
            {item.target for item in plan.capability_report.placements},
            {"cuda-sm86"},
        )
        manifest = plan.to_dict()
        self.assertEqual(
            manifest["capability_analysis"]["target_counts"], {"cuda-sm86": 9}
        )
        self.assertEqual(
            manifest["coverage"]["vortex"]["emitted_stage"],
            "offline-dispatch-plan-only",
        )
        self.assertFalse(
            manifest["coverage"]["vortex"]["executable_binary_emitted"]
        )

    def test_lowers_every_dispatch_to_reviewable_source(self):
        plan = plan_bevformer_graph(_mixed_graph())
        with tempfile.TemporaryDirectory() as directory:
            artifacts = lower_bevformer_dispatch_sources(
                plan, output_directory=directory
            )
            self.assertEqual(len(artifacts), 3)
            self.assertEqual(
                [artifact.artifact_kind for artifact in artifacts],
                [
                    "cuda-source-with-launch-metadata",
                    "standard-tensor-linalg-mlir",
                    "cuda-source-with-launch-metadata",
                ],
            )
            for artifact in artifacts:
                self.assertTrue(artifact.source_sha256)
                self.assertGreater(artifact.source_bytes, 0)
                self.assertFalse(artifact.executable_binary)
                self.assertTrue(Path(artifact.source_path).is_file())
            mlir = Path(artifacts[1].source_path).read_text(encoding="utf-8")
            self.assertEqual(mlir.count("linalg.matmul"), 2)

    def test_complete_1053_node_step_has_real_dag_buffers_and_truthful_coverage(self):
        config = small_step_config()
        encoder_weights, decoder_weights = make_step_weights(config)
        plan = build_bevformer_whole_model_plan(
            encoder_weights,
            decoder_weights,
            config,
            name="complete_step_compiler_test",
        )

        self.assertEqual(len(plan.graph.nodes), 1053)
        self.assertEqual(len(plan.capability_report.regions), 9)
        self.assertEqual(
            set(Counter(strategy.kind for strategy in plan.strategies)),
            {"cnn", "attention", "ffn", "utility"},
        )
        self.assertEqual(
            plan.capability_report.placement("images_to_float32").kind,
            "cnn",
        )
        self.assertEqual(
            plan.capability_report.placement(
                "encoder.geometry.query_visible"
            ).kind,
            "utility",
        )
        self.assertEqual(
            plan.capability_report.placement("encoder.layer0.norm0").kind,
            "attention",
        )
        self.assertEqual(
            plan.capability_report.placement("encoder.layer0.norm1").kind,
            "attention",
        )
        self.assertEqual(
            plan.capability_report.placement("encoder.layer0.norm2").kind,
            "ffn",
        )

        vortex_dispatches = [
            dispatch
            for dispatch in plan.dispatch_dag.dispatches
            if dispatch.target == BEVFORMER_VORTEX_TARGET
        ]
        self.assertEqual(len(vortex_dispatches), 9)
        self.assertTrue(
            all(dispatch.group.startswith("ffn:") for dispatch in vortex_dispatches)
        )
        vortex_graph = plan.vortex_graph_for_dispatch(vortex_dispatches[0].id)
        self.assertEqual(
            tuple(node.op for node in compile_mlir(vortex_graph).graph.nodes),
            ("MatMul", "AddReLU", "MatMul", "Add", "Add", "LayerNorm"),
        )

        expected_values = (
            set(plan.graph.inputs)
            | set(plan.graph.constants)
            | {value for node in plan.graph.nodes for value in node.outputs}
        )
        self.assertEqual(
            {allocation.name for allocation in plan.memory_plan.allocations},
            expected_values,
        )
        self.assertGreater(plan.memory_plan.arena_size, 0)
        self.assertEqual(plan.cross_target_edge_instances, 34)
        self.assertFalse(plan.physical_runtime_ready)
        self.assertFalse(plan.coverage()["full_model_native_vortex"])
        self.assertTrue(plan.coverage()["cuda_fallback_verified"])
        manifest = plan.to_dict()
        self.assertEqual(
            manifest["execution_status"]["cross_target_edge_instances"], 34
        )
        self.assertFalse(manifest["execution_status"]["physical_runtime_ready"])
        json.dumps(manifest, sort_keys=True)


if __name__ == "__main__":
    unittest.main()
