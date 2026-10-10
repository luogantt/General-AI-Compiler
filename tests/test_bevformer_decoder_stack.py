"""Tests for the complete static six-layer BEVFormer decoder stack."""

from __future__ import annotations

import unittest

import numpy as np

from gac.backends.cuda import compile_cuda
from gac.bevformer_decoder import DecoderCrossAttentionConfig
from gac.bevformer_decoder_layer import (
    BevFormerDecoderLayerConfig,
    build_bevformer_decoder_layer_graph,
)
from gac.bevformer_decoder_self import DecoderSelfAttentionConfig
from gac.bevformer_decoder_stack import (
    BevFormerDecoderStackConfig,
    build_bevformer_decoder_stack_graph,
    decoder_stack_weight_shapes,
    validate_decoder_stack_inputs,
)
from gac.bevformer_ffn import BEVFormerFFNConfig
from gac.bevformer_query_init import (
    DecoderQueryInitConfig,
    build_decoder_query_init_graph,
)
from gac.bevformer_refinement import (
    DecoderReferenceRefinementConfig,
    build_decoder_reference_refinement_graph,
)
from gac.errors import ValidationError
from gac.runtime import reference_run


def small_config() -> BevFormerDecoderStackConfig:
    return BevFormerDecoderStackConfig(
        query_init=DecoderQueryInitConfig(queries=3, embed_dim=4),
        layer=BevFormerDecoderLayerConfig(
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
        ),
        refinement=DecoderReferenceRefinementConfig(
            queries=3, embed_dim=4, code_size=6
        ),
    )


def _array_for(name, shape, rng, *, zeros):
    if zeros:
        if name.endswith("norm_weight"):
            return np.ones(shape, dtype=np.float32)
        return np.zeros(shape, dtype=np.float32)
    if name.endswith("norm_weight"):
        return rng.uniform(0.8, 1.2, shape).astype(np.float32)
    if name.endswith("norm_bias"):
        return rng.uniform(-0.1, 0.1, shape).astype(np.float32)
    return (rng.standard_normal(shape) * 0.12).astype(np.float32)


def make_weights(config, seed=8606, *, zeros=False):
    rng = np.random.default_rng(seed)
    inventory = decoder_stack_weight_shapes(config)

    def group(shapes):
        return {
            name: np.ascontiguousarray(
                _array_for(name, shape, rng, zeros=zeros)
            )
            for name, shape in shapes.items()
        }

    return {
        "query_init": group(inventory["query_init"]),
        "self_attention": tuple(
            group(shapes) for shapes in inventory["self_attention"]
        ),
        "cross_attention": tuple(
            group(shapes) for shapes in inventory["cross_attention"]
        ),
        "layer": tuple(group(shapes) for shapes in inventory["layer"]),
        "refinement": tuple(
            group(shapes) for shapes in inventory["refinement"]
        ),
    }


def build_stack(config, weights):
    return build_bevformer_decoder_stack_graph(
        weights["query_init"],
        weights["self_attention"],
        weights["cross_attention"],
        weights["layer"],
        weights["refinement"],
        config,
    )


class DecoderStackTests(unittest.TestCase):
    def test_small_stack_matches_sequential_layer_oracle(self):
        config = small_config()
        weights = make_weights(config)
        graph = build_stack(config, weights)
        rng = np.random.default_rng(3070)
        value = rng.standard_normal((1, 4, 4), dtype=np.float32)
        inputs = {"value": np.ascontiguousarray(value)}
        validate_decoder_stack_inputs(graph, inputs)
        actual = reference_run(graph, inputs)

        initialized = reference_run(
            build_decoder_query_init_graph(
                weights["query_init"], config.query_init
            ),
            {},
        )
        query = initialized["query"]
        query_pos = initialized["query_pos"]
        reference = initialized["reference_points"]
        states = []
        references = []
        regressions = []
        for index in range(config.num_layers):
            layer_graph = build_bevformer_decoder_layer_graph(
                weights["self_attention"][index],
                weights["cross_attention"][index],
                weights["layer"][index],
                config.layer,
            )
            query = reference_run(
                layer_graph,
                {
                    "query": query,
                    "query_pos": query_pos,
                    "value": value,
                    "reference_points": np.ascontiguousarray(
                        reference[..., :2].reshape(1, 3, 1, 2)
                    ),
                },
            )["output"]
            refined = reference_run(
                build_decoder_reference_refinement_graph(
                    weights["refinement"][index], config.refinement
                ),
                {"query": query, "reference_points": reference},
            )
            reference = refined["next_reference_points"]
            states.append(query)
            references.append(reference)
            regressions.append(refined["regression"])

        np.testing.assert_allclose(
            actual["intermediate_states"], np.stack(states), rtol=1e-6, atol=1e-6
        )
        np.testing.assert_allclose(
            actual["initial_reference_points"],
            initialized["reference_points"],
            rtol=1e-6,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            actual["intermediate_reference_points"],
            np.stack(references),
            rtol=1e-6,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            actual["regressions"], np.stack(regressions), rtol=1e-6, atol=1e-6
        )
        np.testing.assert_allclose(
            actual[graph.metadata["final_query_output"]], query, rtol=1e-6, atol=1e-6
        )
        np.testing.assert_allclose(
            actual[graph.metadata["final_reference_output"]],
            reference,
            rtol=1e-6,
            atol=1e-6,
        )

        self.assertTrue(graph.metadata["full_decoder_stack"])
        self.assertEqual(graph.metadata["num_layers"], 6)
        self.assertEqual(
            graph.value_specs["intermediate_states"].shape, (6, 1, 3, 4)
        )
        self.assertEqual(
            graph.value_specs["intermediate_reference_points"].shape,
            (6, 1, 3, 3),
        )
        self.assertEqual(graph.value_specs["regressions"].shape, (6, 1, 3, 6))
        self.assertEqual(
            sum(node.op == "MSDeformableAttention" for node in graph.nodes), 6
        )

    def test_default_tiny_profile_compiles_offline_for_sm86(self):
        config = BevFormerDecoderStackConfig()
        self.assertEqual(
            (
                config.num_layers,
                config.batch,
                config.queries,
                config.embed_dim,
                config.code_size,
            ),
            (6, 1, 900, 256, 10),
        )
        program = compile_cuda(build_stack(config, make_weights(config, zeros=True)))
        description = program.describe()
        self.assertEqual(description["target"], "cuda-sm86")
        self.assertEqual(
            program.reference_graph.value_specs["intermediate_states"].shape,
            (6, 1, 900, 256),
        )
        self.assertEqual(
            program.reference_graph.value_specs[
                "intermediate_reference_points"
            ].shape,
            (6, 1, 900, 3),
        )
        self.assertEqual(
            sum(
                node.op == "MSDeformableAttention"
                for node in program.reference_graph.nodes
            ),
            6,
        )

    def test_strict_profile_weights_and_runtime_input_validation(self):
        config = small_config()
        weights = make_weights(config)
        graph = build_stack(config, weights)
        valid = {"value": np.zeros((1, 4, 4), dtype=np.float32)}
        validate_decoder_stack_inputs(graph, valid)

        with self.assertRaises(ValidationError):
            BevFormerDecoderStackConfig(num_layers=5)
        with self.assertRaisesRegex(ValidationError, "queries"):
            BevFormerDecoderStackConfig(
                query_init=DecoderQueryInitConfig(queries=4, embed_dim=4),
                layer=config.layer,
                refinement=config.refinement,
            )
        with self.assertRaises(ValidationError):
            decoder_stack_weight_shapes(object())  # type: ignore[arg-type]

        with self.assertRaisesRegex(ValidationError, "exactly 6"):
            build_bevformer_decoder_stack_graph(
                weights["query_init"],
                weights["self_attention"][:-1],
                weights["cross_attention"],
                weights["layer"],
                weights["refinement"],
                config,
            )
        bad_query = dict(weights["query_init"])
        bad_query["reference_bias"] = np.zeros((2,), dtype=np.float32)
        with self.assertRaisesRegex(ValidationError, "reference_bias"):
            build_bevformer_decoder_stack_graph(
                bad_query,
                weights["self_attention"],
                weights["cross_attention"],
                weights["layer"],
                weights["refinement"],
                config,
            )

        bad_inputs = {"value": valid["value"].copy()}
        bad_inputs["value"].flat[0] = np.nan
        with self.assertRaisesRegex(ValidationError, "contiguous and finite"):
            validate_decoder_stack_inputs(graph, bad_inputs)
        with self.assertRaises(ValidationError):
            validate_decoder_stack_inputs(graph, {})


if __name__ == "__main__":
    unittest.main()
