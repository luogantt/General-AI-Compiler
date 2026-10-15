from __future__ import annotations

from dataclasses import replace
import unittest

import numpy as np

from gac.backends.mlir import compile_mlir
from gac.bevformer_ffn import BEVFormerFFNConfig, build_bevformer_ffn_graph
from gac.bevformer_partition import (
    BEVFORMER_CUDA_TARGET,
    BEVFORMER_VORTEX_TARGET,
    analyze_bevformer_partitions,
    classify_bevformer_node,
    discover_bevformer_ffn_regions,
    extract_bevformer_ffn_region,
)
from gac.ir import Graph, Node, TensorSpec
from gac.passes import AddReLUFusionPass, ShapeInferencePass


def _weights(channels: int = 4) -> dict[str, np.ndarray]:
    hidden = channels * 2
    rng = np.random.default_rng(1207)
    return {
        "fc1_weight": rng.normal(size=(channels, hidden)).astype(np.float32),
        "fc1_bias": rng.normal(size=(hidden,)).astype(np.float32),
        "fc2_weight": rng.normal(size=(hidden, channels)).astype(np.float32),
        "fc2_bias": rng.normal(size=(channels,)).astype(np.float32),
        "norm_weight": rng.normal(size=(channels,)).astype(np.float32),
        "norm_bias": rng.normal(size=(channels,)).astype(np.float32),
    }


def _ffn_graph() -> Graph:
    return build_bevformer_ffn_graph(
        _weights(),
        BEVFormerFFNConfig(bev_h=1, bev_w=3, embed_dim=4),
        name="partition_test_ffn",
    )


def _rename_node_ids(graph: Graph, ids: tuple[str, ...]) -> Graph:
    renamed = graph.clone()
    renamed.nodes = [
        Node(
            id=node_id,
            op=node.op,
            inputs=node.inputs,
            outputs=node.outputs,
            attrs=dict(node.attrs),
            source_nodes=node.source_nodes,
        )
        for node_id, node in zip(ids, renamed.nodes, strict=True)
    ]
    renamed.validate()
    return renamed


def _wrapped_graph() -> Graph:
    ffn = _ffn_graph()
    graph = Graph(
        name="wrapped_ffn",
        inputs={"x": TensorSpec((1, 3, 4), "float32")},
        outputs=("final",),
        nodes=[
            Node("prepare", "Identity", ("x",), ("query",)),
            *ffn.nodes,
            Node("consume", "Identity", ("output",), ("final",)),
        ],
        constants={name: value.copy() for name, value in ffn.constants.items()},
    )
    ShapeInferencePass().run(graph)
    graph.validate()
    return graph


class BevFormerPartitionTests(unittest.TestCase):
    def test_independent_ffn_extracts_exact_vortex_abi(self):
        graph = _ffn_graph()
        regions = discover_bevformer_ffn_regions(graph)
        self.assertEqual(len(regions), 1)
        region = regions[0]
        self.assertEqual(len(region.node_ids), 7)
        self.assertEqual(region.input_shape, (1, 3, 4))
        self.assertEqual(region.hidden_dim, 8)

        extracted = extract_bevformer_ffn_region(graph, region)
        self.assertEqual(tuple(extracted.inputs), ("query",))
        self.assertEqual(extracted.outputs, ("output",))
        self.assertEqual(
            tuple(extracted.constants),
            (
                "fc1_weight",
                "fc1_bias",
                "fc2_weight",
                "fc2_bias",
                "norm_weight",
                "norm_bias",
            ),
        )
        self.assertEqual(extracted.metadata["kind"], "bevformer-transformer-ffn")
        program = compile_mlir(extracted)
        self.assertEqual(
            tuple(node.op for node in program.graph.nodes),
            ("MatMul", "AddReLU", "MatMul", "Add", "Add", "LayerNorm"),
        )
        self.assertEqual(
            tuple(
                (node.op, node.inputs, node.outputs[0])
                for node in program.graph.nodes
            ),
            (
                ("MatMul", ("query", "fc1_weight"), "fc1_mm"),
                ("AddReLU", ("fc1_mm", "fc1_bias"), "activated"),
                ("MatMul", ("activated", "fc2_weight"), "fc2_mm"),
                ("Add", ("fc2_mm", "fc2_bias"), "projected"),
                ("Add", ("query", "projected"), "ffn_output"),
                (
                    "LayerNorm",
                    ("ffn_output", "norm_weight", "norm_bias"),
                    "output",
                ),
            ),
        )

    def test_encoder_and_decoder_names_are_both_discovered(self):
        encoder_source = _ffn_graph().clone()
        encoder_source.nodes[1] = replace(
            encoder_source.nodes[1],
            inputs=tuple(reversed(encoder_source.nodes[1].inputs)),
        )
        encoder_source.nodes[4] = replace(
            encoder_source.nodes[4],
            inputs=tuple(reversed(encoder_source.nodes[4].inputs)),
        )
        encoder_source.nodes[5] = replace(
            encoder_source.nodes[5],
            inputs=tuple(reversed(encoder_source.nodes[5].inputs)),
        )
        encoder = _rename_node_ids(
            encoder_source,
            (
                "encoder.layer0.ffn_fc1",
                "encoder.layer0.ffn_fc1_bias_add",
                "encoder.layer0.ffn_relu",
                "encoder.layer0.ffn_fc2",
                "encoder.layer0.ffn_fc2_bias_add",
                "encoder.layer0.ffn_residual_add",
                "encoder.layer0.norm2",
            ),
        )
        decoder = _rename_node_ids(
            _ffn_graph(),
            tuple(
                "decoder.decoder.layer0.transformer.ffn." + suffix
                for suffix in (
                    "fc1_matmul",
                    "fc1_bias_add",
                    "relu",
                    "fc2_matmul",
                    "fc2_bias_add",
                    "residual_add",
                    "post_layer_norm",
                )
            ),
        )
        encoder_region = discover_bevformer_ffn_regions(encoder)[0]
        self.assertEqual(encoder_region.region_id, "encoder.layer0.ffn")
        encoder_program = compile_mlir(
            extract_bevformer_ffn_region(encoder, encoder_region)
        )
        self.assertEqual(
            encoder_program.graph.nodes[4].inputs, ("query", "projected")
        )
        self.assertEqual(
            discover_bevformer_ffn_regions(decoder)[0].region_id,
            "decoder.decoder.layer0.transformer.ffn",
        )

    def test_post_fusion_ffn_is_recognized_and_normalized_for_vortex(self):
        graph = _ffn_graph().clone()
        report = AddReLUFusionPass().run(graph)
        self.assertTrue(report.changed)
        graph.validate()

        regions = discover_bevformer_ffn_regions(graph)
        self.assertEqual(len(regions), 1)
        self.assertTrue(regions[0].fused_bias_relu)
        self.assertEqual(len(regions[0].node_ids), 6)

        extracted = extract_bevformer_ffn_region(graph, regions[0])
        self.assertEqual(
            tuple(node.op for node in extracted.nodes),
            ("MatMul", "AddReLU", "MatMul", "Add", "Add", "LayerNorm"),
        )
        compiled = compile_mlir(extracted)
        self.assertEqual(
            tuple(node.op for node in compiled.graph.nodes),
            ("MatMul", "AddReLU", "MatMul", "Add", "Add", "LayerNorm"),
        )

    def test_ffn_inside_larger_graph_has_explicit_mixed_placement(self):
        graph = _wrapped_graph()
        report = analyze_bevformer_partitions(graph)
        self.assertEqual(len(report.regions), 1)
        self.assertEqual(report.placement("prepare").target, BEVFORMER_CUDA_TARGET)
        self.assertEqual(report.placement("consume").target, BEVFORMER_CUDA_TARGET)
        for node_id in report.regions[0].node_ids:
            placement = report.placement(node_id)
            self.assertEqual(placement.kind, "ffn")
            self.assertEqual(placement.target, BEVFORMER_VORTEX_TARGET)
            self.assertEqual(placement.region_id, report.regions[0].region_id)
            self.assertIsNone(placement.reason)

    def test_wrong_residual_falls_back_to_cuda_with_reason(self):
        graph = _ffn_graph().clone()
        residual = graph.nodes[5]
        graph.nodes[5] = replace(
            residual,
            inputs=(residual.inputs[1], residual.inputs[1]),
        )
        ShapeInferencePass().run(graph)
        graph.validate()

        report = analyze_bevformer_partitions(graph)
        self.assertEqual(report.regions, ())
        self.assertTrue(report.rejected_ffn_candidates)
        self.assertIn("original query", report.rejected_ffn_candidates[0][1])
        ffn_placements = [
            placement for placement in report.placements if placement.kind == "ffn"
        ]
        self.assertTrue(ffn_placements)
        self.assertTrue(
            all(placement.target == BEVFORMER_CUDA_TARGET for placement in ffn_placements)
        )
        self.assertTrue(all(placement.reason for placement in ffn_placements))

    def test_external_intermediate_consumer_rejects_region(self):
        graph = _ffn_graph().clone()
        graph.nodes.append(Node("leak", "Identity", ("fc1_mm",), ("leaked",)))
        graph.outputs = ("output", "leaked")
        ShapeInferencePass().run(graph)
        graph.validate()

        report = analyze_bevformer_partitions(graph)
        self.assertEqual(report.regions, ())
        self.assertIn("one internal consumer", report.rejected_ffn_candidates[0][1])

        exposed = _ffn_graph().clone()
        exposed.outputs = ("output", "fc1_mm")
        exposed.validate()
        exposed_report = analyze_bevformer_partitions(exposed)
        self.assertEqual(exposed_report.regions, ())
        self.assertIn("graph outputs", exposed_report.rejected_ffn_candidates[0][1])

    def test_illegal_layer_norm_contract_falls_back_to_cuda_with_reason(self):
        cases = (
            ({"axis": 1, "epsilon": 1e-5}, "last normalization axis"),
            ({"axis": -1, "epsilon": 0.0}, "finite and positive"),
            (
                {"axis": -1, "epsilon": 1e-5, "unsupported": True},
                "unsupported attributes",
            ),
        )
        for attrs, expected_reason in cases:
            with self.subTest(attrs=attrs):
                graph = _ffn_graph().clone()
                graph.nodes[-1] = replace(graph.nodes[-1], attrs=attrs)
                ShapeInferencePass().run(graph)
                graph.validate()

                report = analyze_bevformer_partitions(graph)
                self.assertEqual(report.regions, ())
                self.assertIn(expected_reason, report.rejected_ffn_candidates[0][1])
                ffn_placements = [
                    placement
                    for placement in report.placements
                    if placement.kind == "ffn"
                ]
                self.assertTrue(ffn_placements)
                self.assertTrue(
                    all(
                        placement.target == BEVFORMER_CUDA_TARGET
                        and placement.reason
                        for placement in ffn_placements
                    )
                )
                self.assertTrue(
                    any(
                        expected_reason in placement.reason
                        for placement in ffn_placements
                    )
                )

    def test_rv32_oversized_extent_falls_back_to_cuda_with_reason(self):
        graph = _ffn_graph().clone()
        graph.inputs["query"] = TensorSpec((1, 268_435_456, 4), "float32")
        ShapeInferencePass().run(graph)
        graph.validate()

        report = analyze_bevformer_partitions(graph)
        self.assertEqual(report.regions, ())
        reason = report.rejected_ffn_candidates[0][1]
        self.assertIn("RV32", reason)
        self.assertIn("byte size", reason)
        ffn_placements = [
            placement for placement in report.placements if placement.kind == "ffn"
        ]
        self.assertTrue(ffn_placements)
        self.assertTrue(
            all(
                placement.target == BEVFORMER_CUDA_TARGET
                and placement.reason
                for placement in ffn_placements
            )
        )
        self.assertTrue(
            any(placement.reason == reason for placement in ffn_placements)
        )

    def test_nonfinite_constant_falls_back_before_mlir_lowering(self):
        graph = _ffn_graph().clone()
        graph.constants["fc1_bias"][0] = np.nan

        report = analyze_bevformer_partitions(graph)

        self.assertEqual(report.regions, ())
        reason = report.rejected_ffn_candidates[0][1]
        self.assertIn("finite float32 values", reason)
        self.assertTrue(
            all(
                placement.target == BEVFORMER_CUDA_TARGET
                for placement in report.placements
                if placement.kind == "ffn"
            )
        )

    def test_classifier_prioritizes_ffn_inside_transformer_scope(self):
        ffn = Node(
            "decoder.transformer.attention.ffn.fc1",
            "MatMul",
            ("x", "w"),
            ("y",),
        )
        attention = Node(
            "decoder.cross_attention.softmax",
            "Softmax",
            ("x",),
            ("y",),
        )
        cnn = Node("backbone.layer1.conv", "Conv2D", ("x", "w"), ("y",))
        utility = Node("reshape", "Reshape", ("x",), ("y",), {"shape": [1]})
        self.assertEqual(classify_bevformer_node(ffn), "ffn")
        self.assertEqual(classify_bevformer_node(attention), "attention")
        self.assertEqual(classify_bevformer_node(cnn), "cnn")
        self.assertEqual(classify_bevformer_node(utility), "utility")


if __name__ == "__main__":
    unittest.main()
