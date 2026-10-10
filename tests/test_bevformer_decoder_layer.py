"""Composition tests for one complete BEVFormer decoder layer."""

from __future__ import annotations

import unittest

import numpy as np

from gac.backends.cuda import compile_cuda
from gac.bevformer_decoder import (
    DecoderCrossAttentionConfig,
    build_decoder_cross_attention_graph,
)
from gac.bevformer_decoder_layer import (
    BevFormerDecoderLayerConfig,
    build_bevformer_decoder_layer_graph,
    decoder_layer_weight_shapes,
    validate_decoder_layer_inputs,
)
from gac.bevformer_decoder_self import (
    DecoderSelfAttentionConfig,
    build_decoder_self_attention_graph,
)
from gac.bevformer_ffn import BEVFormerFFNConfig, build_bevformer_ffn_graph
from gac.errors import ValidationError
from gac.runtime import reference_run


def small_config() -> BevFormerDecoderLayerConfig:
    return BevFormerDecoderLayerConfig(
        self_attention=DecoderSelfAttentionConfig(
            queries=3, embed_dim=4, heads=2
        ),
        cross_attention=DecoderCrossAttentionConfig(
            queries=3,
            bev_h=2,
            bev_w=2,
            embed_dim=4,
            heads=2,
            sampling_points=2,
        ),
        ffn=BEVFormerFFNConfig(bev_h=1, bev_w=3, embed_dim=4),
    )


def make_weight_groups(config, seed=811):
    rng = np.random.default_rng(seed)
    result = {}
    for group, shapes in decoder_layer_weight_shapes(config).items():
        values = {}
        for name, shape in shapes.items():
            if "norm" in name and name.endswith("weight"):
                values[name] = rng.uniform(0.8, 1.2, shape).astype(np.float32)
            elif "norm" in name and name.endswith("bias"):
                values[name] = rng.uniform(-0.1, 0.1, shape).astype(np.float32)
            else:
                values[name] = (rng.standard_normal(shape) * 0.12).astype(np.float32)
        result[group] = values
    return result


def make_inputs(config, seed=821):
    rng = np.random.default_rng(seed)
    query_shape = (config.batch, config.queries, config.embed_dim)
    cross = config.cross_attention
    return {
        "query": rng.standard_normal(query_shape, dtype=np.float32),
        "query_pos": rng.standard_normal(query_shape, dtype=np.float32),
        "value": rng.standard_normal(
            (config.batch, cross.spatial_size, config.embed_dim), dtype=np.float32
        ),
        "reference_points": rng.uniform(
            0.1,
            0.9,
            (config.batch, config.queries, cross.levels, 2),
        ).astype(np.float32),
    }


def layer_norm(value, weight, bias, epsilon):
    mean = np.mean(value, axis=-1, keepdims=True)
    variance = np.mean((value - mean) ** 2, axis=-1, keepdims=True)
    return (value - mean) / np.sqrt(variance + np.float32(epsilon)) * weight + bias


def ffn_weights(layer):
    return {
        "fc1_weight": layer["ffn_fc1_weight"],
        "fc1_bias": layer["ffn_fc1_bias"],
        "fc2_weight": layer["ffn_fc2_weight"],
        "fc2_bias": layer["ffn_fc2_bias"],
        "norm_weight": layer["norm2_weight"],
        "norm_bias": layer["norm2_bias"],
    }


class DecoderLayerProfileTests(unittest.TestCase):
    def test_default_profile_operation_order_and_wiring(self):
        config = BevFormerDecoderLayerConfig()
        groups = make_weight_groups(config)
        graph = build_bevformer_decoder_layer_graph(
            groups["self_attention"], groups["cross_attention"], groups["layer"], config
        )

        self.assertEqual(graph.inputs["query"].shape, (1, 900, 256))
        self.assertEqual(graph.inputs["value"].shape, (1, 2500, 256))
        self.assertEqual(graph.inputs["reference_points"].shape, (1, 900, 1, 2))
        self.assertEqual(graph.value_specs["output"].shape, (1, 900, 256))
        self.assertEqual(
            graph.metadata["operation_order"],
            ["self_attn", "norm0", "cross_attn", "norm1", "ffn", "norm2"],
        )
        self.assertEqual(graph.metadata["scope"], "one-decoder-transformer-layer")
        self.assertFalse(graph.metadata["full_decoder"])
        nodes = {node.id: node for node in graph.nodes}
        self.assertEqual(
            nodes["cross_attention.query_position_add"].inputs,
            ("norm0_output", "query_pos"),
        )
        self.assertEqual(
            nodes["cross_attention.residual_add"].inputs,
            ("cross_attention.projected", "norm0_output"),
        )
        self.assertEqual(nodes["norm1"].inputs[0], "cross_attention_output")
        self.assertEqual(
            nodes["ffn.residual_add"].inputs,
            ("norm1_output", "ffn.projected"),
        )
        self.assertEqual(nodes["ffn.post_layer_norm"].outputs, ("output",))

    def test_default_profile_compiles_offline(self):
        config = BevFormerDecoderLayerConfig()
        groups = make_weight_groups(config)
        program = compile_cuda(
            build_bevformer_decoder_layer_graph(
                groups["self_attention"],
                groups["cross_attention"],
                groups["layer"],
                config,
            )
        )
        self.assertEqual(len(program.launches), 41)
        self.assertEqual(sum(node.op == "AddReLU" for node in program.graph.nodes), 1)
        self.assertEqual(len(program.describe()["zero_copy_views"]), 8)
        self.assertEqual(program.reference_graph.value_specs["output"].shape, (1, 900, 256))
        self.assertEqual(program.describe()["target"], "cuda-sm86")
        self.assertEqual(sum(node.op == "MSDeformableAttention" for node in program.graph.nodes), 1)

    def test_profile_contract_rejects_component_mismatches_and_missing_norms(self):
        base = small_config()
        cases = [
            {"self_attention": DecoderSelfAttentionConfig(queries=4, embed_dim=4, heads=2)},
            {"cross_attention": DecoderCrossAttentionConfig(queries=3, bev_h=2, bev_w=2, embed_dim=8, heads=2)},
            {"cross_attention": DecoderCrossAttentionConfig(queries=3, bev_h=2, bev_w=2, embed_dim=4, heads=1)},
            {"ffn": BEVFormerFFNConfig(bev_h=1, bev_w=2, embed_dim=4)},
            {"self_attention": DecoderSelfAttentionConfig(queries=3, embed_dim=4, heads=2, post_layer_norm=False)},
            {"ffn": BEVFormerFFNConfig(bev_h=1, bev_w=3, embed_dim=4, post_layer_norm=False)},
            {"norm1_epsilon": 0.0},
            {"norm1_epsilon": float("nan")},
            {"norm1_epsilon": 1e-50},
        ]
        for overrides in cases:
            arguments = {
                "self_attention": base.self_attention,
                "cross_attention": base.cross_attention,
                "ffn": base.ffn,
                "norm1_epsilon": base.norm1_epsilon,
            }
            arguments.update(overrides)
            with self.subTest(overrides=overrides), self.assertRaises(ValidationError):
                BevFormerDecoderLayerConfig(**arguments)


class DecoderLayerReferenceTests(unittest.TestCase):
    def setUp(self):
        self.config = small_config()
        self.groups = make_weight_groups(self.config)
        self.inputs = make_inputs(self.config)
        self.graph = build_bevformer_decoder_layer_graph(
            self.groups["self_attention"],
            self.groups["cross_attention"],
            self.groups["layer"],
            self.config,
        )

    def test_composition_matches_stage_by_stage_reference_run_oracle(self):
        self_graph = build_decoder_self_attention_graph(
            self.groups["self_attention"], self.config.self_attention
        )
        norm0 = reference_run(
            self_graph,
            {"query": self.inputs["query"], "query_pos": self.inputs["query_pos"]},
        )["output"]

        cross_graph = build_decoder_cross_attention_graph(
            self.groups["cross_attention"], self.config.cross_attention
        )
        cross = reference_run(
            cross_graph,
            {
                "query": norm0,
                "query_pos": self.inputs["query_pos"],
                "value": self.inputs["value"],
                "reference_points": self.inputs["reference_points"],
            },
        )["output"]
        layer = self.groups["layer"]
        norm1 = layer_norm(
            cross,
            layer["norm1_weight"],
            layer["norm1_bias"],
            self.config.norm1_epsilon,
        )

        ffn_graph = build_bevformer_ffn_graph(
            ffn_weights(layer), self.config.ffn
        )
        expected = reference_run(ffn_graph, {"query": norm1})["output"]

        validate_decoder_layer_inputs(self.graph, self.inputs)
        actual = reference_run(self.graph, self.inputs)["output"]
        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)

    def test_graph_owns_all_three_weight_group_copies(self):
        before = self.graph.constants["self_attention.attention.q_weight"].copy()
        self.groups["self_attention"]["q_weight"].fill(77.0)
        np.testing.assert_array_equal(
            self.graph.constants["self_attention.attention.q_weight"], before
        )
        before = self.graph.constants["cross_attention.value_weight"].copy()
        self.groups["cross_attention"]["value_weight"].fill(66.0)
        np.testing.assert_array_equal(
            self.graph.constants["cross_attention.value_weight"], before
        )
        before = self.graph.constants["ffn.fc1_weight"].copy()
        self.groups["layer"]["ffn_fc1_weight"].fill(55.0)
        np.testing.assert_array_equal(self.graph.constants["ffn.fc1_weight"], before)


class DecoderLayerContractTests(unittest.TestCase):
    def setUp(self):
        self.config = small_config()
        self.groups = make_weight_groups(self.config)

    def test_each_weight_group_rejects_names_types_shapes_layout_and_values(self):
        mutations = []
        missing = dict(self.groups["self_attention"])
        missing.pop("q_bias")
        mutations.append(("self_attention", missing))
        non_string = dict(self.groups["cross_attention"])
        non_string[1] = np.zeros((1,), np.float32)
        mutations.append(("cross_attention", non_string))
        not_array = dict(self.groups["layer"])
        not_array["norm1_bias"] = [0.0] * 4
        mutations.append(("layer", not_array))
        wrong_shape = dict(self.groups["layer"])
        wrong_shape["ffn_fc1_weight"] = np.zeros((4, 7), np.float32)
        mutations.append(("layer", wrong_shape))
        wrong_dtype = dict(self.groups["cross_attention"])
        wrong_dtype["value_bias"] = wrong_dtype["value_bias"].astype(np.float64)
        mutations.append(("cross_attention", wrong_dtype))
        noncontiguous = dict(self.groups["self_attention"])
        noncontiguous["q_weight"] = np.zeros((4, 8), np.float32)[:, ::2]
        mutations.append(("self_attention", noncontiguous))
        nonfinite = dict(self.groups["layer"])
        nonfinite["norm2_weight"] = nonfinite["norm2_weight"].copy()
        nonfinite["norm2_weight"][0] = np.inf
        mutations.append(("layer", nonfinite))

        for scope, broken in mutations:
            groups = {name: dict(values) for name, values in self.groups.items()}
            groups[scope] = broken
            with self.subTest(scope=scope), self.assertRaises(ValidationError):
                build_bevformer_decoder_layer_graph(
                    groups["self_attention"],
                    groups["cross_attention"],
                    groups["layer"],
                    self.config,
                )

    def test_input_contract_rejects_names_types_shapes_layout_values_and_range(self):
        graph = build_bevformer_decoder_layer_graph(
            self.groups["self_attention"],
            self.groups["cross_attention"],
            self.groups["layer"],
            self.config,
        )
        valid = make_inputs(self.config)
        cases = []
        missing = dict(valid)
        missing.pop("query_pos")
        cases.append(missing)
        not_array = dict(valid)
        not_array["query"] = not_array["query"].tolist()
        cases.append(not_array)
        wrong_shape = dict(valid)
        wrong_shape["value"] = np.zeros((1, 3, 4), np.float32)
        cases.append(wrong_shape)
        wrong_dtype = dict(valid)
        wrong_dtype["query_pos"] = wrong_dtype["query_pos"].astype(np.float64)
        cases.append(wrong_dtype)
        noncontiguous = dict(valid)
        noncontiguous["query"] = np.zeros((1, 3, 8), np.float32)[:, :, ::2]
        cases.append(noncontiguous)
        nonfinite = dict(valid)
        nonfinite["value"] = nonfinite["value"].copy()
        nonfinite["value"].flat[0] = np.nan
        cases.append(nonfinite)
        out_of_range = dict(valid)
        out_of_range["reference_points"] = out_of_range["reference_points"].copy()
        out_of_range["reference_points"].flat[0] = np.float32(1.01)
        cases.append(out_of_range)

        for index, inputs in enumerate(cases):
            with self.subTest(index=index), self.assertRaises(ValidationError):
                validate_decoder_layer_inputs(graph, inputs)

    def test_builder_rejects_invalid_arguments(self):
        with self.assertRaises(ValidationError):
            decoder_layer_weight_shapes(object())  # type: ignore[arg-type]
        with self.assertRaises(ValidationError):
            build_bevformer_decoder_layer_graph(
                self.groups["self_attention"],
                self.groups["cross_attention"],
                self.groups["layer"],
                object(),  # type: ignore[arg-type]
            )
        with self.assertRaises(ValidationError):
            build_bevformer_decoder_layer_graph(
                self.groups["self_attention"],
                self.groups["cross_attention"],
                self.groups["layer"],
                self.config,
                name="",
            )


if __name__ == "__main__":
    unittest.main()
