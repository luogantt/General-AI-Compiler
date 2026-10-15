from __future__ import annotations

import json
import unittest

import numpy as np

from gac.dispatch_runtime import (
    DispatchRuntimePlan,
    build_dispatch_runtime_plan,
    execute_dispatch_plan,
)
from gac.errors import ValidationError
from gac.ir import Graph, Node, TensorSpec
from gac.partition import NodePlacement, partition_graph


def _branched_graph() -> Graph:
    spec = TensorSpec((2,), "float32")
    graph = Graph(
        name="branched_dispatch_runtime",
        inputs={"x": spec},
        outputs=("y",),
        nodes=[
            Node("prepare", "Identity", ("x",), ("stem",)),
            Node("left", "Relu", ("stem",), ("left_value",)),
            Node("right", "Add", ("stem", "bias"), ("right_value",)),
            Node(
                "merge",
                "Add",
                ("left_value", "right_value"),
                ("sum",),
            ),
            Node("finish", "Identity", ("sum",), ("y",)),
        ],
        constants={"bias": np.array([10.0, 10.0], dtype=np.float32)},
        value_specs={
            "x": spec,
            "bias": spec,
            "stem": spec,
            "left_value": spec,
            "right_value": spec,
            "sum": spec,
            "y": spec,
        },
    )
    graph.validate()
    return graph


def _runtime_plan():
    graph = _branched_graph()
    dag = partition_graph(
        graph,
        {
            "prepare": "cuda-sm86",
            "left": "cuda-sm86",
            "right": "vortex-v3",
            "merge": "cuda-sm86",
            "finish": "cuda-sm86",
        },
    )
    return build_dispatch_runtime_plan(dag, alignment=64)


def _returning_target_plan():
    spec = TensorSpec((2,), "float32")
    graph = Graph(
        name="returning_target",
        inputs={"x": spec},
        outputs=("y",),
        nodes=[
            Node("first", "Add", ("x", "bias"), ("first_out",)),
            Node("middle", "Identity", ("first_out",), ("middle_out",)),
            Node("last", "Add", ("middle_out", "bias"), ("y",)),
        ],
        constants={"bias": np.ones((2,), dtype=np.float32)},
        value_specs={
            "first_out": spec,
            "middle_out": spec,
            "y": spec,
        },
    )
    dag = partition_graph(
        graph,
        {"first": "target-a", "middle": "target-b", "last": "target-a"},
    )
    return build_dispatch_runtime_plan(dag, alignment=64)


def _same_target_fanout_plan():
    spec = TensorSpec((2,), "float32")
    graph = Graph(
        name="same_target_fanout",
        inputs={"x": spec},
        outputs=("y",),
        nodes=[
            Node("produce", "Identity", ("x",), ("shared",)),
            Node("left", "Add", ("shared", "bias"), ("left_out",)),
            Node("right", "Relu", ("shared",), ("right_out",)),
            Node("merge", "Add", ("left_out", "right_out"), ("y",)),
        ],
        constants={"bias": np.array([10.0, 10.0], dtype=np.float32)},
        value_specs={
            "x": spec,
            "bias": spec,
            "shared": spec,
            "left_out": spec,
            "right_out": spec,
            "y": spec,
        },
    )
    dag = partition_graph(
        graph,
        {
            "produce": NodePlacement("target-a", "producer"),
            "left": NodePlacement("target-b", "left"),
            "right": NodePlacement("target-b", "right"),
            "merge": NodePlacement("target-a", "merge"),
        },
    )
    return build_dispatch_runtime_plan(dag, alignment=64)


class DispatchRuntimePlanTests(unittest.TestCase):
    def test_whole_graph_memory_is_projected_onto_dispatches(self):
        plan = _runtime_plan()
        self.assertEqual(
            tuple(step.dispatch_id for step in plan.steps),
            ("dispatch_0000", "dispatch_0001", "dispatch_0002"),
        )
        self.assertEqual(plan.steps[0].dependencies, ())
        self.assertEqual(plan.steps[1].dependencies, ("dispatch_0000",))
        self.assertEqual(
            plan.steps[2].dependencies,
            ("dispatch_0000", "dispatch_0001"),
        )

        first, second, third = plan.steps
        self.assertEqual(
            {binding.value: binding.role for binding in first.bindings},
            {
                "x": "input",
                "stem": "output",
                "left_value": "output",
            },
        )
        self.assertEqual(
            {binding.value: binding.role for binding in second.bindings},
            {
                "stem": "input",
                "right_value": "output",
                "bias": "constant",
            },
        )
        self.assertEqual(
            {binding.value: binding.role for binding in third.bindings},
            {
                "left_value": "input",
                "right_value": "input",
                "y": "output",
                "sum": "temporary",
            },
        )

        stem_output = next(
            binding for binding in first.outputs if binding.value == "stem"
        )
        stem_input = next(
            binding for binding in second.inputs if binding.value == "stem"
        )
        self.assertEqual(stem_output.allocation, stem_input.allocation)
        self.assertEqual(stem_output.allocation.storage, "arena")
        self.assertEqual(stem_output.allocation.alignment, 64)
        self.assertEqual(
            second.constants[0].allocation.storage,
            "resident-constant",
        )
        self.assertEqual(third.outputs[0].allocation.storage, "external-output")

        self.assertIn("stem", second.release_after)
        self.assertIn("left_value", third.release_after)
        self.assertIn("right_value", third.release_after)
        self.assertIn("sum", third.release_after)

    def test_plan_serialization_is_deterministic_and_truthfully_labelled(self):
        first = _runtime_plan().to_dict()
        second = _runtime_plan().to_dict()
        self.assertEqual(first, second)
        self.assertEqual(first["kind"], "host-dispatch-orchestration-plan")
        self.assertEqual(first["memory_plan"]["graph_name"], first["dispatch_dag"]["graph"])
        self.assertEqual(
            [step["dispatch_id"] for step in first["steps"]],
            first["dispatch_dag"]["schedule"],
        )

    def test_global_boundary_plan_excludes_dispatch_private_storage(self):
        plan = _runtime_plan()

        self.assertEqual(
            tuple(item.name for item in plan.global_memory_plan.allocations),
            ("x", "bias", "stem", "left_value", "right_value", "y"),
        )
        # sum is consumed only inside the final dispatch.  The whole-graph
        # reference plan retains it, while target lowering owns its physical
        # register/LMEM allocation.
        self.assertEqual(plan.memory_plan.allocation("sum").storage, "arena")
        with self.assertRaises(KeyError):
            plan.global_memory_plan.allocation("sum")

    def test_runtime_schedule_makes_every_cross_target_copy_explicit(self):
        plan = _runtime_plan()

        self.assertEqual(len(plan.setup_steps), 1)
        self.assertEqual(
            [
                (step.value, step.source_target, step.target, step.phase)
                for step in plan.transfer_steps
            ],
            [
                ("bias", "host", "vortex-v3", "setup"),
                ("x", "host", "cuda-sm86", "invoke"),
                ("stem", "cuda-sm86", "vortex-v3", "invoke"),
                ("right_value", "vortex-v3", "cuda-sm86", "invoke"),
                ("y", "cuda-sm86", "host", "invoke"),
            ],
        )
        self.assertEqual(
            tuple(
                step.dispatch_id
                for step in plan.runtime_steps
                if step.kind == "dispatch"
            ),
            plan.dag.schedule,
        )
        positions = {step.id: index for index, step in enumerate(plan.runtime_steps)}
        for step in plan.runtime_steps:
            self.assertTrue(
                all(positions[dependency] < positions[step.id] for dependency in step.depends_on)
            )

    def test_buffer_lifetimes_use_runtime_step_coordinates_and_targets(self):
        plan = _runtime_plan()
        lifetimes = {item.name: item for item in plan.buffer_lifetimes}

        self.assertEqual(
            (lifetimes["stem"].first_step, lifetimes["stem"].last_step),
            (2, 4),
        )
        self.assertEqual(lifetimes["stem"].targets, ("cuda-sm86", "vortex-v3"))
        self.assertEqual(
            (lifetimes["right_value"].first_step, lifetimes["right_value"].last_step),
            (4, 6),
        )
        self.assertEqual(
            lifetimes["right_value"].targets,
            ("cuda-sm86", "vortex-v3"),
        )
        self.assertEqual(
            (lifetimes["y"].first_step, lifetimes["y"].last_step),
            (6, 7),
        )
        self.assertEqual(lifetimes["y"].targets, ("cuda-sm86", "host"))

    def test_manifest_round_trip_rebuilds_from_authoritative_graph(self):
        plan = _runtime_plan()
        serialized = json.loads(plan.to_json())

        restored = DispatchRuntimePlan.from_dict(
            serialized, source_graph=plan.dag.source_graph
        )
        self.assertEqual(restored.to_dict(), plan.to_dict())
        self.assertEqual(
            DispatchRuntimePlan.from_json(
                plan.to_json(indent=None), source_graph=plan.dag.source_graph
            ).to_dict(),
            plan.to_dict(),
        )

        serialized["runtime_steps"][-1]["source_target"] = "wrong-target"
        with self.assertRaisesRegex(ValidationError, "does not match"):
            DispatchRuntimePlan.from_dict(
                serialized, source_graph=plan.dag.source_graph
            )

    def test_immutable_constant_upload_is_reused_when_target_reappears(self):
        plan = _returning_target_plan()
        uploads = [
            step
            for step in plan.setup_steps
            if step.kind == "transfer" and step.value == "bias"
        ]

        self.assertEqual(len(uploads), 1)
        self.assertEqual(uploads[0].target, "target-a")
        final_dispatch = next(
            step for step in plan.runtime_steps if step.dispatch_id == "dispatch_0002"
        )
        self.assertIn(uploads[0].id, final_dispatch.depends_on)

    def test_host_scheduler_requires_explicit_cross_target_adapters(self):
        plan = _runtime_plan()
        calls: list[str] = []

        def unused(dispatch, inputs, constants):
            calls.append(dispatch.id)
            return {}

        with self.assertRaisesRegex(ValidationError, "transfer adapters"):
            execute_dispatch_plan(
                plan,
                {"x": np.array([1.0, 2.0], dtype=np.float32)},
                {dispatch.id: unused for dispatch in plan.dag.dispatches},
            )
        self.assertEqual(calls, [])

    def test_host_scheduler_executes_dependencies_and_adapters_in_order(self):
        plan = _runtime_plan()
        calls: list[str] = []
        transfers: list[tuple[str, str, str]] = []

        def first(dispatch, inputs, constants):
            self.assertEqual(constants, {})
            calls.append(dispatch.id)
            stem = inputs["x"] + np.float32(1.0)
            return {"stem": stem, "left_value": stem * np.float32(2.0)}

        def second(dispatch, inputs, constants):
            calls.append(dispatch.id)
            return {"right_value": inputs["stem"] + constants["bias"]}

        def third(dispatch, inputs, constants):
            self.assertEqual(constants, {})
            calls.append(dispatch.id)
            return {"y": inputs["left_value"] + inputs["right_value"]}

        def copy_adapter(value, edge, producer, consumer):
            transfers.append((edge.value, producer.target, consumer.target))
            return value.copy()

        result = execute_dispatch_plan(
            plan,
            {"x": np.array([1.0, 2.0], dtype=np.float32)},
            {
                "dispatch_0000": first,
                "dispatch_0001": second,
                "dispatch_0002": third,
            },
            transfer_adapters={
                ("cuda-sm86", "vortex-v3"): copy_adapter,
                ("vortex-v3", "cuda-sm86"): copy_adapter,
            },
        )
        self.assertEqual(calls, ["dispatch_0000", "dispatch_0001", "dispatch_0002"])
        self.assertEqual(
            transfers,
            [
                ("stem", "cuda-sm86", "vortex-v3"),
                ("right_value", "vortex-v3", "cuda-sm86"),
            ],
        )
        np.testing.assert_array_equal(
            result.outputs["y"],
            np.array([16.0, 19.0], dtype=np.float32),
        )
        self.assertEqual(
            tuple(record.dispatch_id for record in result.records),
            plan.dag.schedule,
        )
        self.assertEqual(result.records[1].adapted_inputs, ("stem",))
        self.assertEqual(result.records[2].adapted_inputs, ("right_value",))

    def test_host_scheduler_reuses_one_transfer_for_same_target_fanout(self):
        plan = _same_target_fanout_plan()
        shared_transfers = [
            step
            for step in plan.transfer_steps
            if step.value == "shared" and step.target == "target-b"
        ]
        self.assertEqual(len(shared_transfers), 1)

        adapter_calls: list[tuple[str, str, str, str]] = []

        def adapter(value, edge, producer, consumer):
            adapter_calls.append(
                (edge.value, producer.target, consumer.target, consumer.id)
            )
            return value.copy()

        result = execute_dispatch_plan(
            plan,
            {"x": np.array([1.0, 2.0], dtype=np.float32)},
            {
                "dispatch_0000": lambda dispatch, inputs, constants: {
                    "shared": inputs["x"] + np.float32(1.0)
                },
                "dispatch_0001": lambda dispatch, inputs, constants: {
                    "left_out": inputs["shared"] + constants["bias"]
                },
                "dispatch_0002": lambda dispatch, inputs, constants: {
                    "right_out": inputs["shared"] * np.float32(2.0)
                },
                "dispatch_0003": lambda dispatch, inputs, constants: {
                    "y": inputs["left_out"] + inputs["right_out"]
                },
            },
            transfer_adapters={
                ("target-a", "target-b"): adapter,
                ("target-b", "target-a"): adapter,
            },
        )

        self.assertEqual(
            [call for call in adapter_calls if call[0] == "shared"],
            [("shared", "target-a", "target-b", "dispatch_0001")],
        )
        self.assertEqual(result.records[1].adapted_inputs, ("shared",))
        self.assertEqual(result.records[2].adapted_inputs, ())
        np.testing.assert_array_equal(
            result.outputs["y"], np.array([16.0, 19.0], dtype=np.float32)
        )

    def test_executor_output_contract_is_checked(self):
        plan = _runtime_plan()

        def wrong(dispatch, inputs, constants):
            return {"wrong": inputs["x"]}

        def adapter(value, edge, producer, consumer):
            return value

        with self.assertRaisesRegex(ValidationError, "incorrect outputs"):
            execute_dispatch_plan(
                plan,
                {"x": np.array([1.0, 2.0], dtype=np.float32)},
                {
                    "dispatch_0000": wrong,
                    "dispatch_0001": wrong,
                    "dispatch_0002": wrong,
                },
                transfer_adapters={
                    ("cuda-sm86", "vortex-v3"): adapter,
                    ("vortex-v3", "cuda-sm86"): adapter,
                },
            )


if __name__ == "__main__":
    unittest.main()
