"""Shared pipeline order and Vortex's actual boundaries, artifacts and ABI."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from gac.compiler.pipeline import COMPILER_STAGES, pipeline_manifest, run_pipeline
from gac.core.errors import BackendError
from gac.core.ir import Graph, Node, TensorSpec
from gac.targets.vortex import compiler as vortex


def graph() -> Graph:
    return Graph("pipeline_chain", {"x": TensorSpec((2, 3), "float32")}, ("y",), [
        Node("add", "Add", ("x", "bias"), ("a",)),
        Node("relu", "ReLU", ("a",), ("b",)),
        Node("twice", "Add", ("b", "b"), ("c",)),
        Node("output", "Sub", ("c", "bias"), ("y",)),
    ], constants={"bias": np.asarray([0.1, -0.2, 0.3], np.float32)})


class CompilerPipelineTests(unittest.TestCase):
    def test_catalog_is_ordered_and_manifest_is_not_mutable_shared_state(self):
        self.assertEqual([s.number for s in COMPILER_STAGES], list(range(1, 11)))
        self.assertEqual(len({s.name for s in COMPILER_STAGES}), 10)
        self.assertEqual(len({s.artifact for s in COMPILER_STAGES}), 10)
        manifest = pipeline_manifest()
        manifest[0]["name"] = "changed"
        self.assertEqual(pipeline_manifest()[0]["name"], "model-import")

    def test_runner_uses_catalog_order_and_returns_each_stage_result(self):
        handlers = {stage.name: (lambda state, name=stage.name: [*state, name])
                    for stage in reversed(COMPILER_STAGES)}
        self.assertEqual(run_pipeline([], handlers), [s.name for s in COMPILER_STAGES])

    def test_missing_extra_or_noncallable_handlers_fail_before_any_work(self):
        visited = []
        handlers = {stage.name: (lambda state: visited.append(True) or state)
                    for stage in COMPILER_STAGES}
        for bad in ({k: v for k, v in handlers.items() if k != "runtime"},
                    {**handlers, "extra-stage": lambda state: state}):
            with self.assertRaisesRegex(ValueError, "handlers mismatch"):
                run_pipeline(None, bad)
        with self.assertRaisesRegex(TypeError, "callable"):
            run_pipeline(None, {**handlers, "runtime": None})
        self.assertEqual(visited, [])

    def test_failed_stage_does_not_execute_later_layers(self):
        visited = []
        def work(state, name):
            visited.append(name)
            if name == "legalization":
                raise BackendError("unsupported target operation")
            return state
        handlers = {s.name: (lambda state, name=s.name: work(state, name)) for s in COMPILER_STAGES}
        with self.assertRaisesRegex(BackendError, "unsupported target"):
            run_pipeline(None, handlers)
        self.assertEqual(visited, [s.name for s in COMPILER_STAGES[:4]])


class VortexPipelineTests(unittest.TestCase):
    def test_template_tiling_schedule_codegen_and_runtime_are_actual_separate_steps(self):
        names = ["stage_model_import", "stage_ir_conversion", "stage_graph_optimization",
                 "stage_legalization", "stage_subgraph_partition", "stage_template_selection",
                 "stage_tiling", "stage_schedule", "stage_codegen", "stage_runtime"]
        visited = []
        snapshots = {}
        def wrapper(name, original):
            def invoke(state):
                visited.append(name)
                result = original(state)
                snapshots[name] = {"partitions": len(state.partitions), "templates": len(state.templates),
                                   "tiles": len(state.tiles), "dispatches": len(state.dispatches),
                                   "memory": state.memory_plan is not None,
                                   "device": bool(state.source), "host": bool(state.host_source)}
                return result
            return invoke
        with mock.patch.multiple(vortex, **{name: wrapper(name, getattr(vortex, name)) for name in names}):
            program = vortex.compile_vortex_graph(graph())
        self.assertEqual(visited, names)
        self.assertEqual(snapshots["stage_subgraph_partition"],
                         {"partitions": 4, "templates": 0, "tiles": 0, "dispatches": 0,
                          "memory": False, "device": False, "host": False})
        self.assertEqual(snapshots["stage_template_selection"]["templates"], 4)
        self.assertEqual(snapshots["stage_template_selection"]["tiles"], 0)
        self.assertEqual(snapshots["stage_tiling"]["tiles"], 4)
        self.assertEqual(snapshots["stage_tiling"]["dispatches"], 0)
        self.assertTrue(snapshots["stage_schedule"]["memory"])
        self.assertEqual(snapshots["stage_schedule"]["dispatches"], 4)
        self.assertFalse(snapshots["stage_schedule"]["device"])
        self.assertTrue(snapshots["stage_codegen"]["device"])
        self.assertFalse(snapshots["stage_codegen"]["host"])
        self.assertTrue(snapshots["stage_runtime"]["host"])
        self.assertEqual([d["wait_for_dispatch"] for d in program.dispatches], [None, 0, 1, 2])
        self.assertEqual(program.memory_plan.allocation("a").offset,
                         program.memory_plan.allocation("c").offset)

    def test_program_and_emitted_stage_artifacts_share_one_catalog(self):
        program = vortex.compile_vortex_graph(graph())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = vortex.write_vortex_graph_artifact(root, program)
            self.assertEqual(program.stages, pipeline_manifest())
            self.assertEqual(report["stages"], pipeline_manifest())
            for stage in COMPILER_STAGES:
                payload = (root / stage.artifact).read_text(encoding="utf-8")
                if stage.name == "codegen":
                    self.assertEqual(payload, program.source)
                else:
                    self.assertIsNotNone(json.loads(payload))
            self.assertEqual((root / "main.cpp").read_text(encoding="utf-8"), program.host_source)


if __name__ == "__main__":
    unittest.main()
