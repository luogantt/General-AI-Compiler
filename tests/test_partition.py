from __future__ import annotations

import json
import unittest

import numpy as np

from gac.errors import BackendError, ValidationError
from gac.ir import Graph, Node, TensorSpec
from gac.partition import (
    BackendCapability,
    DispatchDAG,
    NodePlacement,
    RegionAssignment,
    assign_backends,
    extract_subgraph,
    partition_by_capabilities,
    partition_graph,
    partition_regions,
)


class PartitionTests(unittest.TestCase):
    def graph(self) -> Graph:
        spec = TensorSpec((2, 2), "float32")
        graph = Graph(
            name="branched_bevformer",
            inputs={"image": spec, "context": spec},
            outputs=("output",),
            nodes=[
                Node("conv", "Conv2D", ("image", "weight"), ("cnn0",)),
                Node("relu", "ReLU", ("cnn0",), ("cnn1",)),
                Node("attention", "Attention", ("cnn1", "context"), ("attn",)),
                Node("ffn", "MatMul", ("attn", "weight"), ("projected",)),
                Node("skip", "Identity", ("cnn1",), ("residual",)),
                Node("join", "Add", ("projected", "residual"), ("output",)),
            ],
            constants={"weight": np.eye(2, dtype=np.float32)},
            value_specs={
                "cnn0": spec,
                "cnn1": spec,
                "attn": spec,
                "projected": spec,
                "residual": spec,
                "output": spec,
            },
        )
        graph.validate()
        return graph

    def capabilities(self) -> tuple[BackendCapability, ...]:
        return (
            BackendCapability(
                "vortex-v3",
                frozenset({"MatMul"}),
                check=lambda node, _: None if node.id == "ffn" else "not an FFN",
                group="ffn",
            ),
            BackendCapability(
                "cuda-sm86", frozenset({"Conv2D", "ReLU"}), group="cnn"
            ),
            BackendCapability(
                "cuda-sm86", frozenset({"Attention"}), group="attention"
            ),
            BackendCapability(
                "cuda-sm86", frozenset({"Identity", "Add"}), group="utility"
            ),
        )

    def test_ordered_capabilities_create_semantic_dispatches(self) -> None:
        graph = self.graph()
        placements = assign_backends(graph, self.capabilities())
        self.assertEqual(
            list(placements.items()),
            [
                ("conv", NodePlacement("cuda-sm86", "cnn")),
                ("relu", NodePlacement("cuda-sm86", "cnn")),
                ("attention", NodePlacement("cuda-sm86", "attention")),
                ("ffn", NodePlacement("vortex-v3", "ffn")),
                ("skip", NodePlacement("cuda-sm86", "utility")),
                ("join", NodePlacement("cuda-sm86", "utility")),
            ],
        )

        dag = partition_by_capabilities(graph, self.capabilities())
        self.assertEqual(
            [(item.target, item.group, item.node_ids) for item in dag.dispatches],
            [
                ("cuda-sm86", "cnn", ("conv", "relu")),
                ("cuda-sm86", "attention", ("attention",)),
                ("vortex-v3", "ffn", ("ffn",)),
                ("cuda-sm86", "utility", ("skip", "join")),
            ],
        )
        self.assertEqual(dag.dependencies("dispatch_0003"), ("dispatch_0000", "dispatch_0002"))

        cnn = dag.dispatch("dispatch_0000")
        self.assertEqual(cnn.inputs, ("image",))
        self.assertEqual(cnn.outputs, ("cnn1",))
        self.assertEqual(tuple(cnn.graph.constants), ("weight",))
        utility = dag.dispatch("dispatch_0003")
        self.assertEqual(utility.inputs, ("cnn1", "projected"))
        self.assertEqual(utility.outputs, ("output",))

        edge_values = [edge.value for edge in dag.edges]
        self.assertEqual(
            edge_values,
            ["image", "context", "cnn1", "attn", "projected", "output"],
        )
        self.assertNotIn("weight", edge_values)

    def test_plain_target_mapping_remains_compatible(self) -> None:
        graph = self.graph()
        dag = partition_graph(graph, {node.id: "cuda-sm86" for node in graph.nodes})
        self.assertEqual(len(dag.dispatches), 1)
        self.assertEqual(dag.dispatches[0].node_ids, tuple(node.id for node in graph.nodes))
        self.assertIsNone(dag.dispatches[0].group)

    def test_explicit_regions_preserve_same_target_boundary(self) -> None:
        graph = self.graph()
        dag = partition_regions(
            graph,
            (
                RegionAssignment("cuda-sm86", ("conv", "relu"), "cnn"),
                RegionAssignment("cuda-sm86", ("attention",), "attention"),
                RegionAssignment("vortex-v3", ("ffn",), "ffn"),
                RegionAssignment("cuda-sm86", ("skip", "join"), "utility"),
            ),
        )
        self.assertEqual(len(dag.dispatches), 4)
        with self.assertRaisesRegex(ValidationError, "cover every graph node"):
            partition_regions(
                graph,
                (RegionAssignment("cuda-sm86", ("conv",), "cnn"),),
            )

    def test_subgraph_external_values_and_constants(self) -> None:
        graph = self.graph()
        subgraph = extract_subgraph(graph, ("attention", "ffn"), name="attention_ffn")
        self.assertEqual(tuple(subgraph.inputs), ("cnn1", "context"))
        self.assertEqual(tuple(subgraph.constants), ("weight",))
        self.assertEqual(subgraph.outputs, ("projected",))
        self.assertEqual(tuple(node.id for node in subgraph.nodes), ("attention", "ffn"))
        subgraph.validate()

    def test_capability_priority_and_fallback_are_explicit(self) -> None:
        graph = self.graph()
        generic = BackendCapability("cpu", None, group="fallback")
        placements = assign_backends(graph, (generic, *self.capabilities()))
        self.assertTrue(all(item.target == "cpu" for item in placements.values()))

        placements = assign_backends(
            graph,
            (),
            fallback_target="cpu",
            fallback_group="fallback",
        )
        self.assertEqual(placements["conv"], NodePlacement("cpu", "fallback"))
        with self.assertRaisesRegex(BackendError, "no backend can execute node 'conv'"):
            assign_backends(graph, ())
        with self.assertRaisesRegex(ValidationError, "requires fallback_target"):
            assign_backends(graph, (), fallback_group="fallback")

    def test_manifest_round_trip_reconstructs_from_source_graph(self) -> None:
        graph = self.graph()
        dag = partition_by_capabilities(graph, self.capabilities())
        restored = DispatchDAG.from_json(dag.to_json(), source_graph=graph)
        self.assertEqual(restored.to_dict(), dag.to_dict())

        tampered = dag.to_dict()
        tampered["edges"][2]["producer"] = "dispatch_0002"
        with self.assertRaisesRegex(ValidationError, "edges do not match"):
            DispatchDAG.from_dict(tampered, source_graph=graph)

        tampered = json.loads(dag.to_json())
        tampered["dispatches"][0]["inputs"] = ["context"]
        with self.assertRaisesRegex(ValidationError, "boundary metadata"):
            DispatchDAG.from_dict(tampered, source_graph=graph)

    def test_placement_must_cover_nodes_exactly(self) -> None:
        graph = self.graph()
        with self.assertRaisesRegex(ValidationError, "missing=.*join"):
            partition_graph(
                graph,
                {node.id: "cuda-sm86" for node in graph.nodes[:-1]},
            )


if __name__ == "__main__":
    unittest.main()
