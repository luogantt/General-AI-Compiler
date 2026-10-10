from __future__ import annotations

import unittest

import numpy as np

from gac.bevformer_ffn import BEVFormerFFNConfig, ffn_weight_shapes
from gac.bevformer_spatial import SpatialCrossAttentionConfig
from gac.bevformer_temporal import TemporalSelfAttentionConfig, temporal_weight_shapes
from gac.bevformer_decoder import (
    DecoderCrossAttentionConfig,
    decoder_cross_attention_weight_shapes,
)
from gac.bevformer_refinement import (
    DecoderReferenceRefinementConfig,
    decoder_refinement_weight_shapes,
)
from gac.bevformer_decoder_self import (
    DecoderSelfAttentionConfig,
    decoder_self_attention_weight_shapes,
)
from gac.bevformer_encoder_layer import (
    BevFormerEncoderLayerConfig,
    encoder_layer_weight_shapes,
)
from gac.bevformer_geometry import BevProjectionConfig
from gac.bevformer_query_init import DecoderQueryInitConfig
from gac.errors import ValidationError
from gac.frontends.bevformer_weights import (
    DECODER_PREFIX,
    ENCODER_PREFIX,
    extract_decoder_cross_attention_weights,
    extract_decoder_refinement_weights,
    extract_decoder_query_init_weights,
    extract_decoder_self_attention_weights,
    extract_encoder_ffn_weights,
    extract_encoder_layer_weights,
    extract_spatial_attention_weights,
    extract_temporal_attention_weights,
)


def add_linear(state, prefix, input_channels, output_channels, start):
    values = np.arange(
        start,
        start + input_channels * output_channels,
        dtype=np.float32,
    ).reshape(output_channels, input_channels)
    state[f"{prefix}.weight"] = values
    state[f"{prefix}.bias"] = np.arange(output_channels, dtype=np.float32)
    return values


class BevFormerWeightMappingTests(unittest.TestCase):
    def test_temporal_mapping_uses_official_names_and_transposes_linear_weights(self):
        config = TemporalSelfAttentionConfig(
            bev_h=2, bev_w=2, embed_dim=4, heads=2, sampling_points=2
        )
        base = f"{ENCODER_PREFIX}.1.attentions.0"
        state = {}
        original = {}
        shapes = temporal_weight_shapes(config)
        for index, (canonical, module) in enumerate(
            (
                ("value", "value_proj"),
                ("offset", "sampling_offsets"),
                ("attention", "attention_weights"),
                ("output", "output_proj"),
            )
        ):
            inputs, outputs = shapes[f"{canonical}_weight"]
            original[canonical] = add_linear(
                state, f"{base}.{module}", inputs, outputs, index * 1000
            )
        mapped = extract_temporal_attention_weights(state, 1, config)
        self.assertEqual(set(mapped), set(shapes))
        for canonical in original:
            np.testing.assert_array_equal(
                mapped[f"{canonical}_weight"], original[canonical].T
            )
            self.assertTrue(mapped[f"{canonical}_weight"].flags.c_contiguous)

    def test_spatial_mapping_distinguishes_outer_and_nested_output_projection(self):
        config = SpatialCrossAttentionConfig(
            cameras=2,
            queries=4,
            embed_dim=4,
            heads=2,
            spatial_shapes=((2, 2),),
            sampling_points=4,
            z_anchors=2,
        )
        outer = f"{ENCODER_PREFIX}.0.attentions.1"
        nested = f"{outer}.deformable_attention"
        state = {}
        value = add_linear(state, f"{nested}.value_proj", 4, 4, 0)
        offsets = add_linear(state, f"{nested}.sampling_offsets", 4, 16, 100)
        attention = add_linear(state, f"{nested}.attention_weights", 4, 8, 200)
        output = add_linear(state, f"{outer}.output_proj", 4, 4, 300)
        # This tensor mirrors the nested module's intentionally unused field;
        # the adapter must select the active outer projection instead.
        add_linear(state, f"{nested}.output_proj", 4, 4, 400)
        mapped = extract_spatial_attention_weights(state, 0, config)
        np.testing.assert_array_equal(mapped["value_weight"], value.T)
        np.testing.assert_array_equal(mapped["offset_weight"], offsets.T)
        np.testing.assert_array_equal(mapped["attention_weight"], attention.T)
        np.testing.assert_array_equal(mapped["output_weight"], output.T)

    def test_encoder_ffn_mapping_uses_official_names_and_post_norm(self):
        config = BEVFormerFFNConfig(bev_h=1, bev_w=2, embed_dim=4)
        base = f"{ENCODER_PREFIX}.3"
        state = {}
        fc1 = add_linear(state, f"{base}.ffns.0.layers.0.0", 4, 8, 100)
        fc2 = add_linear(state, f"{base}.ffns.0.layers.1", 8, 4, 200)
        norm_weight = np.linspace(0.5, 1.0, 4, dtype=np.float32)
        norm_bias = np.linspace(-0.2, 0.2, 4, dtype=np.float32)
        state[f"{base}.norms.2.weight"] = norm_weight
        state[f"{base}.norms.2.bias"] = norm_bias

        mapped = extract_encoder_ffn_weights(state, 3, config)

        self.assertEqual(set(mapped), set(ffn_weight_shapes(config)))
        np.testing.assert_array_equal(mapped["fc1_weight"], fc1.T)
        np.testing.assert_array_equal(mapped["fc2_weight"], fc2.T)
        np.testing.assert_array_equal(mapped["norm_weight"], norm_weight)
        np.testing.assert_array_equal(mapped["norm_bias"], norm_bias)
        self.assertTrue(mapped["fc1_weight"].flags.c_contiguous)
        self.assertTrue(mapped["fc2_weight"].flags.c_contiguous)
        self.assertIsNot(mapped["norm_weight"], norm_weight)
        self.assertIsNot(mapped["norm_bias"], norm_bias)

    def test_encoder_ffn_without_post_norm_does_not_read_norms(self):
        config = BEVFormerFFNConfig(
            bev_h=1, bev_w=1, embed_dim=2, post_layer_norm=False
        )
        base = f"{ENCODER_PREFIX}.0"
        state = {}
        add_linear(state, f"{base}.ffns.0.layers.0.0", 2, 4, 0)
        add_linear(state, f"{base}.ffns.0.layers.1", 4, 2, 100)

        mapped = extract_encoder_ffn_weights(state, 0, config)

        self.assertEqual(set(mapped), set(ffn_weight_shapes(config)))
        self.assertNotIn("norm_weight", mapped)
        self.assertNotIn("norm_bias", mapped)

    def test_complete_encoder_layer_mapping_matches_group_contract(self):
        temporal = TemporalSelfAttentionConfig(
            bev_h=1, bev_w=2, embed_dim=4, heads=2, sampling_points=2
        )
        spatial = SpatialCrossAttentionConfig(
            cameras=2,
            queries=2,
            embed_dim=4,
            heads=2,
            spatial_shapes=((1, 2),),
            sampling_points=4,
            z_anchors=2,
        )
        config = BevFormerEncoderLayerConfig(
            temporal=temporal,
            spatial=spatial,
            projection=BevProjectionConfig(
                cameras=2, bev_h=1, bev_w=2, z_anchors=2
            ),
            ffn_hidden_dim=6,
        )
        base = f"{ENCODER_PREFIX}.1"
        state = {}
        temporal_shapes = temporal_weight_shapes(temporal)
        for index, (canonical, module) in enumerate(
            (
                ("value", "value_proj"),
                ("offset", "sampling_offsets"),
                ("attention", "attention_weights"),
                ("output", "output_proj"),
            )
        ):
            inputs, outputs = temporal_shapes[f"{canonical}_weight"]
            add_linear(state, f"{base}.attentions.0.{module}", inputs, outputs, index * 100)
        spatial_inner = f"{base}.attentions.1.deformable_attention"
        add_linear(state, f"{spatial_inner}.value_proj", 4, 4, 500)
        add_linear(state, f"{spatial_inner}.sampling_offsets", 4, 16, 600)
        add_linear(state, f"{spatial_inner}.attention_weights", 4, 8, 700)
        add_linear(state, f"{base}.attentions.1.output_proj", 4, 4, 800)
        add_linear(state, f"{base}.ffns.0.layers.0.0", 4, 6, 900)
        add_linear(state, f"{base}.ffns.0.layers.1", 6, 4, 1000)
        for index in range(3):
            state[f"{base}.norms.{index}.weight"] = np.full((4,), index + 1, np.float32)
            state[f"{base}.norms.{index}.bias"] = np.full((4,), -index, np.float32)
        mapped = extract_encoder_layer_weights(state, 1, config)
        inventory = encoder_layer_weight_shapes(config)
        self.assertEqual(set(mapped), set(inventory))
        for group in inventory:
            self.assertEqual(set(mapped[group]), set(inventory[group]))
        np.testing.assert_array_equal(
            mapped["layer"]["norm1_weight"], np.full((4,), 2, np.float32)
        )

    def test_encoder_ffn_mapping_rejects_invalid_contracts(self):
        config = BEVFormerFFNConfig(bev_h=1, bev_w=1, embed_dim=2)
        base = f"{ENCODER_PREFIX}.0"
        state = {}
        add_linear(state, f"{base}.ffns.0.layers.0.0", 2, 4, 0)
        add_linear(state, f"{base}.ffns.0.layers.1", 4, 2, 100)
        state[f"{base}.norms.2.weight"] = np.ones((2,), np.float32)
        state[f"{base}.norms.2.bias"] = np.zeros((2,), np.float32)

        invalid_arguments = [
            (state, -1, config, ENCODER_PREFIX),
            (state, True, config, ENCODER_PREFIX),
            (state, 0, object(), ENCODER_PREFIX),
            (state, 0, config, ""),
            ([], 0, config, ENCODER_PREFIX),
            ({1: np.zeros((1,), np.float32)}, 0, config, ENCODER_PREFIX),
        ]
        for weights, layer, candidate_config, prefix in invalid_arguments:
            with self.subTest(
                layer=layer, config=type(candidate_config).__name__, prefix=prefix
            ), self.assertRaises(ValidationError):
                extract_encoder_ffn_weights(
                    weights,  # type: ignore[arg-type]
                    layer,
                    candidate_config,  # type: ignore[arg-type]
                    encoder_prefix=prefix,
                )

        mutations = (
            (f"{base}.ffns.0.layers.0.0.weight", "missing"),
            (f"{base}.ffns.0.layers.0.0.weight", "shape"),
            (f"{base}.ffns.0.layers.1.bias", "dtype"),
            (f"{base}.ffns.0.layers.1.weight", "layout"),
            (f"{base}.norms.2.weight", "nonfinite"),
            (f"{base}.norms.2.bias", "not-array"),
        )
        for key, mutation in mutations:
            broken = dict(state)
            if mutation == "missing":
                del broken[key]
            elif mutation == "shape":
                broken[key] = np.zeros((3, 2), np.float32)
            elif mutation == "dtype":
                broken[key] = broken[key].astype(np.float64)
            elif mutation == "layout":
                broken[key] = np.zeros((2, 8), np.float32)[:, ::2]
                self.assertEqual(broken[key].shape, (2, 4))
                self.assertFalse(broken[key].flags.c_contiguous)
            elif mutation == "nonfinite":
                broken[key] = broken[key].copy()
                broken[key][0] = np.inf
            else:
                broken[key] = [0.0, 0.0]
            with self.subTest(mutation=mutation), self.assertRaises(ValidationError):
                extract_encoder_ffn_weights(broken, 0, config)

    def test_mapping_rejects_missing_shape_dtype_layout_nonfinite_and_bad_layer(self):
        config = TemporalSelfAttentionConfig(
            bev_h=1, bev_w=1, embed_dim=2, heads=1, sampling_points=1
        )
        base = f"{ENCODER_PREFIX}.0.attentions.0"
        state = {}
        shapes = temporal_weight_shapes(config)
        for canonical, module in (
            ("value", "value_proj"),
            ("offset", "sampling_offsets"),
            ("attention", "attention_weights"),
            ("output", "output_proj"),
        ):
            inputs, outputs = shapes[f"{canonical}_weight"]
            add_linear(state, f"{base}.{module}", inputs, outputs, 0)
        with self.assertRaises(ValidationError):
            extract_temporal_attention_weights(state, -1, config)
        for mutation in ("missing", "shape", "dtype", "layout", "nonfinite"):
            broken = dict(state)
            key = f"{base}.value_proj.weight"
            if mutation == "missing":
                del broken[key]
            elif mutation == "shape":
                broken[key] = np.zeros((3, 2), np.float32)
            elif mutation == "dtype":
                broken[key] = broken[key].astype(np.float64)
            elif mutation == "layout":
                broken[key] = np.zeros((2, 4), np.float32)[:, ::2]
            else:
                broken[key] = broken[key].copy()
                broken[key][0, 0] = np.nan
            with self.subTest(mutation=mutation), self.assertRaises(ValidationError):
                extract_temporal_attention_weights(broken, 0, config)

    def test_decoder_cross_attention_mapping_uses_second_attention_slot(self):
        config = DecoderCrossAttentionConfig(
            queries=3, bev_h=2, bev_w=2, embed_dim=4, heads=2, sampling_points=2
        )
        base = f"{DECODER_PREFIX}.5.attentions.1"
        state = {}
        originals = {}
        shapes = decoder_cross_attention_weight_shapes(config)
        for index, (canonical, module) in enumerate(
            (
                ("value", "value_proj"),
                ("offset", "sampling_offsets"),
                ("attention", "attention_weights"),
                ("output", "output_proj"),
            )
        ):
            inputs, outputs = shapes[f"{canonical}_weight"]
            originals[canonical] = add_linear(
                state, f"{base}.{module}", inputs, outputs, 5000 + index * 1000
            )
        mapped = extract_decoder_cross_attention_weights(state, 5, config)
        self.assertEqual(set(mapped), set(shapes))
        for canonical, original in originals.items():
            np.testing.assert_array_equal(mapped[f"{canonical}_weight"], original.T)

    def test_decoder_refinement_mapping_uses_official_regression_branch(self):
        config = DecoderReferenceRefinementConfig(
            queries=3, embed_dim=4, code_size=6
        )
        base = "pts_bbox_head.reg_branches.5"
        state = {}
        originals = {}
        shapes = decoder_refinement_weight_shapes(config)
        for index, (canonical, module) in enumerate(
            (("fc0", "0"), ("fc1", "2"), ("fc2", "4"))
        ):
            inputs, outputs = shapes[f"{canonical}_weight"]
            originals[canonical] = add_linear(
                state, f"{base}.{module}", inputs, outputs, 8000 + index * 1000
            )
        mapped = extract_decoder_refinement_weights(state, 5, config)
        self.assertEqual(set(mapped), set(shapes))
        for canonical, original in originals.items():
            np.testing.assert_array_equal(mapped[f"{canonical}_weight"], original.T)

    def test_decoder_self_attention_mapping_splits_packed_qkv_and_norm0(self):
        config = DecoderSelfAttentionConfig(
            queries=3, embed_dim=4, heads=2
        )
        base = f"{DECODER_PREFIX}.2"
        attention = f"{base}.attentions.0.attn"
        packed_weight = np.arange(48, dtype=np.float32).reshape(12, 4)
        packed_bias = np.arange(12, dtype=np.float32)
        state = {
            f"{attention}.in_proj_weight": packed_weight,
            f"{attention}.in_proj_bias": packed_bias,
            f"{attention}.out_proj.weight": np.arange(16, dtype=np.float32).reshape(4, 4),
            f"{attention}.out_proj.bias": np.arange(4, dtype=np.float32),
            f"{base}.norms.0.weight": np.linspace(0.5, 1.0, 4, dtype=np.float32),
            f"{base}.norms.0.bias": np.linspace(-0.1, 0.1, 4, dtype=np.float32),
        }
        mapped = extract_decoder_self_attention_weights(state, 2, config)
        self.assertEqual(set(mapped), set(decoder_self_attention_weight_shapes(config)))
        for index, canonical in enumerate(("q", "k", "v")):
            start, end = index * 4, (index + 1) * 4
            np.testing.assert_array_equal(
                mapped[f"{canonical}_weight"], packed_weight[start:end].T
            )
            np.testing.assert_array_equal(
                mapped[f"{canonical}_bias"], packed_bias[start:end]
            )
            self.assertTrue(mapped[f"{canonical}_weight"].flags.c_contiguous)

    def test_decoder_query_initialization_mapping(self):
        config = DecoderQueryInitConfig(queries=3, embed_dim=4)
        embedding = np.arange(24, dtype=np.float32).reshape(3, 8)
        reference_weight = np.arange(12, dtype=np.float32).reshape(3, 4)
        reference_bias = np.arange(3, dtype=np.float32)
        state = {
            "pts_bbox_head.query_embedding.weight": embedding,
            "pts_bbox_head.transformer.reference_points.weight": reference_weight,
            "pts_bbox_head.transformer.reference_points.bias": reference_bias,
        }
        mapped = extract_decoder_query_init_weights(state, config)
        np.testing.assert_array_equal(mapped["query_embedding"], embedding)
        np.testing.assert_array_equal(
            mapped["reference_weight"], reference_weight.T
        )
        np.testing.assert_array_equal(mapped["reference_bias"], reference_bias)


if __name__ == "__main__":
    unittest.main()
