from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import gac.tools.build_bevformer as build_bevformer_tool
from gac.errors import ValidationError
from gac.frontends.bevformer_backbone import ResNetFpnProfile, required_checkpoint_specs
from gac.package import load_package
from gac.bevformer_spatial import SpatialCrossAttentionConfig
from gac.bevformer_temporal import TemporalSelfAttentionConfig, temporal_weight_shapes
from gac.bevformer_decoder import (
    DecoderCrossAttentionConfig,
    decoder_cross_attention_weight_shapes,
)
from gac.bevformer_ffn import BEVFormerFFNConfig
from gac.bevformer_decoder_self import DecoderSelfAttentionConfig
from gac.bevformer_decoder_layer import BevFormerDecoderLayerConfig
from gac.bevformer_detection_head import DetectionHeadConfig
from gac.bevformer_refinement import DecoderReferenceRefinementConfig
from gac.bevformer_encoder_layer import BevFormerEncoderLayerConfig
from gac.bevformer_encoder_stack import BevFormerEncoderStackConfig
from gac.bevformer_geometry import BevProjectionConfig
from gac.bevformer_query_init import DecoderQueryInitConfig
from gac.bevformer_feature_prep import BevFormerFeaturePrepConfig
from gac.bevformer_encoder_inputs import BevFormerEncoderInputConfig
from gac.bevformer_decoder_stack import BevFormerDecoderStackConfig
from gac.bevformer_detection_stack import DetectionStackConfig
from gac.bevformer_decoder_pipeline import BevFormerDecoderPipelineConfig
from gac.bevformer_encoder_pipeline import BevFormerColdEncoderPipelineConfig
from gac.bevformer_cold_pipeline import BevFormerColdPipelineConfig
from gac.bevformer_history_pipeline import BevFormerHistoryPipelineConfig
from gac.bevformer_motion import BevFormerMotionConfig
from gac.bevformer_step_pipeline import BevFormerStepPipelineConfig
from gac.tools.build_bevformer import (
    build_backbone_package,
    build_decoder_cross_attention_package,
    build_decoder_layer_package,
    build_decoder_stack_package,
    build_decoder_refinement_package,
    build_decoder_query_init_package,
    build_decoder_self_attention_package,
    build_detection_head_package,
    build_detection_stack_package,
    build_decoder_pipeline_package,
    build_cold_pipeline_package,
    build_history_encoder_package,
    build_streaming_step_package,
    build_encoder_ffn_package,
    build_encoder_layer_package,
    build_encoder_stack_package,
    build_encoder_inputs_package,
    build_feature_prep_package,
    build_spatial_attention_package,
    build_temporal_attention_package,
    inspect_bevformer_weights,
    build_parser,
)
from gac.frontends.bevformer_weights import (
    extract_decoder_layer_weights,
    extract_decoder_stack_weights,
    extract_decoder_refinement_weights,
    extract_detection_head_weights,
    extract_detection_stack_weights,
    extract_feature_prep_weights,
    extract_encoder_input_weights,
)
from gac.weights import write_weights


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_SHA256 = "7" * 64
SMALL_PROFILE = ResNetFpnProfile(
    name="bevformer-builder-small",
    batch=1,
    input_channels=3,
    input_hw=(16, 16),
    stem_channels=2,
    stage_planes=(2, 3, 4, 5),
    blocks=(1, 1, 1, 1),
    expansion=2,
    fpn_channels=4,
    input_dtype="float32",
)


def make_weights() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(3070)
    result: dict[str, np.ndarray] = {}
    for spec in required_checkpoint_specs(SMALL_PROFILE):
        if spec.role == "bn-running-variance":
            value = rng.uniform(0.5, 1.5, size=spec.shape)
        else:
            value = rng.normal(0, 0.1, size=spec.shape)
        result[spec.key] = np.ascontiguousarray(value, dtype=spec.dtype)
    result["pts_bbox_head.unused"] = np.ones((1,), dtype=np.float32)
    return result


def add_linear(state, prefix, input_channels, output_channels):
    state[f"{prefix}.weight"] = np.zeros(
        (output_channels, input_channels), dtype=np.float32
    )
    state[f"{prefix}.bias"] = np.zeros((output_channels,), dtype=np.float32)


def make_attention_weights():
    temporal = TemporalSelfAttentionConfig(
        bev_h=2, bev_w=2, embed_dim=4, heads=2, sampling_points=2
    )
    spatial = SpatialCrossAttentionConfig(
        cameras=2,
        queries=4,
        embed_dim=4,
        heads=2,
        spatial_shapes=((2, 2),),
        sampling_points=4,
        z_anchors=2,
    )
    state = {}
    temporal_base = "pts_bbox_head.transformer.encoder.layers.0.attentions.0"
    temporal_shapes = temporal_weight_shapes(temporal)
    for canonical, module in (
        ("value", "value_proj"),
        ("offset", "sampling_offsets"),
        ("attention", "attention_weights"),
        ("output", "output_proj"),
    ):
        inputs, outputs = temporal_shapes[f"{canonical}_weight"]
        add_linear(state, f"{temporal_base}.{module}", inputs, outputs)
    spatial_outer = "pts_bbox_head.transformer.encoder.layers.0.attentions.1"
    spatial_inner = f"{spatial_outer}.deformable_attention"
    add_linear(state, f"{spatial_inner}.value_proj", 4, 4)
    add_linear(state, f"{spatial_inner}.sampling_offsets", 4, 16)
    add_linear(state, f"{spatial_inner}.attention_weights", 4, 8)
    add_linear(state, f"{spatial_outer}.output_proj", 4, 4)
    decoder = DecoderCrossAttentionConfig(
        queries=3, bev_h=2, bev_w=2, embed_dim=4, heads=2, sampling_points=2
    )
    decoder_base = "pts_bbox_head.transformer.decoder.layers.0.attentions.1"
    decoder_shapes = decoder_cross_attention_weight_shapes(decoder)
    for canonical, module in (
        ("value", "value_proj"),
        ("offset", "sampling_offsets"),
        ("attention", "attention_weights"),
        ("output", "output_proj"),
    ):
        inputs, outputs = decoder_shapes[f"{canonical}_weight"]
        add_linear(state, f"{decoder_base}.{module}", inputs, outputs)
    ffn = BEVFormerFFNConfig(bev_h=2, bev_w=2, embed_dim=4)
    encoder_base = "pts_bbox_head.transformer.encoder.layers.0"
    add_linear(state, f"{encoder_base}.ffns.0.layers.0.0", 4, 8)
    add_linear(state, f"{encoder_base}.ffns.0.layers.1", 8, 4)
    state[f"{encoder_base}.norms.2.weight"] = np.ones((4,), dtype=np.float32)
    state[f"{encoder_base}.norms.2.bias"] = np.zeros((4,), dtype=np.float32)
    state[f"{encoder_base}.norms.0.weight"] = np.ones((4,), dtype=np.float32)
    state[f"{encoder_base}.norms.0.bias"] = np.zeros((4,), dtype=np.float32)
    state[f"{encoder_base}.norms.1.weight"] = np.ones((4,), dtype=np.float32)
    state[f"{encoder_base}.norms.1.bias"] = np.zeros((4,), dtype=np.float32)
    decoder_self = DecoderSelfAttentionConfig(queries=3, embed_dim=4, heads=2)
    decoder_layer = "pts_bbox_head.transformer.decoder.layers.0"
    decoder_attn = f"{decoder_layer}.attentions.0.attn"
    state[f"{decoder_attn}.in_proj_weight"] = np.zeros((12, 4), dtype=np.float32)
    state[f"{decoder_attn}.in_proj_bias"] = np.zeros((12,), dtype=np.float32)
    add_linear(state, f"{decoder_attn}.out_proj", 4, 4)
    state[f"{decoder_layer}.norms.0.weight"] = np.ones((4,), dtype=np.float32)
    state[f"{decoder_layer}.norms.0.bias"] = np.zeros((4,), dtype=np.float32)
    add_linear(state, f"{decoder_layer}.ffns.0.layers.0.0", 4, 8)
    add_linear(state, f"{decoder_layer}.ffns.0.layers.1", 8, 4)
    for index in (1, 2):
        state[f"{decoder_layer}.norms.{index}.weight"] = np.ones(
            (4,), dtype=np.float32
        )
        state[f"{decoder_layer}.norms.{index}.bias"] = np.zeros(
            (4,), dtype=np.float32
        )
    refinement = DecoderReferenceRefinementConfig(queries=3, embed_dim=4, code_size=6)
    refinement_base = "pts_bbox_head.reg_branches.0"
    add_linear(state, f"{refinement_base}.0", 4, 4)
    add_linear(state, f"{refinement_base}.2", 4, 4)
    add_linear(state, f"{refinement_base}.4", 4, 6)
    query_init = DecoderQueryInitConfig(queries=3, embed_dim=4)
    state["pts_bbox_head.query_embedding.weight"] = np.zeros(
        (3, 8), dtype=np.float32
    )
    add_linear(state, "pts_bbox_head.transformer.reference_points", 4, 3)
    detection = DetectionHeadConfig(
        queries=3,
        embed_dim=4,
        classes=2,
        code_size=6,
        point_cloud_range=(-2.0, -3.0, -1.0, 2.0, 5.0, 3.0),
    )
    classification = "pts_bbox_head.cls_branches.0"
    add_linear(state, f"{classification}.0", 4, 4)
    state[f"{classification}.1.weight"] = np.ones((4,), dtype=np.float32)
    state[f"{classification}.1.bias"] = np.zeros((4,), dtype=np.float32)
    add_linear(state, f"{classification}.3", 4, 4)
    state[f"{classification}.4.weight"] = np.ones((4,), dtype=np.float32)
    state[f"{classification}.4.bias"] = np.zeros((4,), dtype=np.float32)
    add_linear(state, f"{classification}.6", 4, 2)
    feature_prep = BevFormerFeaturePrepConfig(
        cameras=2,
        embed_dim=4,
        feature_h=2,
        feature_w=2,
    )
    state["pts_bbox_head.transformer.cams_embeds"] = np.arange(
        8, dtype=np.float32
    ).reshape(2, 4)
    state["pts_bbox_head.transformer.level_embeds"] = np.arange(
        4, dtype=np.float32
    ).reshape(1, 4)
    encoder_inputs = BevFormerEncoderInputConfig(
        bev_h=2,
        bev_w=2,
        embed_dim=4,
        can_bus_dim=3,
        can_bus_hidden_dim=5,
    )
    state["pts_bbox_head.bev_embedding.weight"] = np.arange(
        16, dtype=np.float32
    ).reshape(4, 4)
    state["pts_bbox_head.positional_encoding.row_embed.weight"] = np.arange(
        4, dtype=np.float32
    ).reshape(2, 2)
    state["pts_bbox_head.positional_encoding.col_embed.weight"] = np.arange(
        4, 8, dtype=np.float32
    ).reshape(2, 2)
    add_linear(state, "pts_bbox_head.transformer.can_bus_mlp.0", 3, 5)
    add_linear(state, "pts_bbox_head.transformer.can_bus_mlp.2", 5, 4)
    state["pts_bbox_head.transformer.can_bus_mlp.norm.weight"] = np.ones(
        (4,), dtype=np.float32
    )
    state["pts_bbox_head.transformer.can_bus_mlp.norm.bias"] = np.zeros(
        (4,), dtype=np.float32
    )

    encoder_layer0 = "pts_bbox_head.transformer.encoder.layers.0"
    for target_layer in (1, 2):
        target_prefix = f"pts_bbox_head.transformer.encoder.layers.{target_layer}"
        for name, value in tuple(state.items()):
            if name.startswith(f"{encoder_layer0}."):
                suffix = name[len(encoder_layer0) :]
                state[f"{target_prefix}{suffix}"] = value.copy()
    decoder_prefixes = (
        "pts_bbox_head.transformer.decoder.layers.0",
        "pts_bbox_head.reg_branches.0",
        "pts_bbox_head.cls_branches.0",
    )
    for target_layer in range(1, 6):
        for source_prefix in decoder_prefixes:
            target_prefix = source_prefix[:-1] + str(target_layer)
            for name, value in tuple(state.items()):
                if name.startswith(f"{source_prefix}."):
                    suffix = name[len(source_prefix) :]
                    state[f"{target_prefix}{suffix}"] = value.copy()
    return (
        state,
        temporal,
        spatial,
        decoder,
        ffn,
        decoder_self,
        refinement,
        query_init,
        detection,
        feature_prep,
        encoder_inputs,
    )


class BevFormerBuildTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.weights = write_weights(
            self.root / "tiny.gacweights",
            make_weights(),
            source_checkpoint_sha256=CHECKPOINT_SHA256,
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_inspect_weights_has_reproducible_inventory_and_provenance(self):
        report = inspect_bevformer_weights(
            self.weights,
            expected_checkpoint_sha256=CHECKPOINT_SHA256.upper(),
        )
        self.assertTrue(report["verified"])
        self.assertEqual(report["source_checkpoint_sha256"], CHECKPOINT_SHA256)
        self.assertEqual(report["dtype_counts"], {"float32": len(make_weights())})
        self.assertEqual(report["top_level_prefix_counts"]["img_backbone"], len(make_weights()) - 5)
        self.assertEqual(report["top_level_prefix_counts"]["pts_bbox_head"], 1)
        with self.assertRaisesRegex(ValidationError, "provenance mismatch"):
            inspect_bevformer_weights(
                self.weights,
                expected_checkpoint_sha256="8" * 64,
            )

    def test_backbone_package_is_roundtrip_verified_and_records_provenance(self):
        output = self.root / "backbone.hxp"
        report = build_backbone_package(
            self.weights,
            output,
            expected_checkpoint_sha256=CHECKPOINT_SHA256,
            profile=SMALL_PROFILE,
        )
        self.assertTrue(report["codegen_verified"])
        self.assertFalse(report["physical_cuda_execution"])
        self.assertEqual(report["model"], "BEVFormer-custom-static")
        self.assertEqual(report["output_spec"]["shape"], [1, 4, 1, 1])
        self.assertEqual(report["source_checkpoint_sha256"], CHECKPOINT_SHA256)
        loaded = load_package(output)
        self.assertEqual(
            loaded.manifest["metadata"]["source_checkpoint_sha256"],
            CHECKPOINT_SHA256,
        )
        self.assertEqual(loaded.manifest["metadata"]["validation_state"], "offline-codegen-and-package-roundtrip")
        self.assertEqual(loaded.graph.metadata["ignored_checkpoint_key_count"], 1)

    def test_cli_inspect_weights_and_suffix_validation(self):
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "gac.tools.build_bevformer",
                "inspect-weights",
                str(self.weights),
                "--expected-checkpoint-sha256",
                CHECKPOINT_SHA256,
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(json.loads(result.stdout)["verified"])
        with self.assertRaisesRegex(ValidationError, "must end in .hxp"):
            build_backbone_package(
                self.weights,
                self.root / "backbone.zip",
                profile=SMALL_PROFILE,
            )

    def test_component_cli_exposes_stack_layer_and_raw_head_commands(self):
        parser = build_parser()
        encoder = parser.parse_args(
            [
                "encoder-stack",
                "--weights",
                "model.gacweights",
                "--mode",
                "cold",
                "-o",
                "encoder.hxp",
            ]
        )
        self.assertEqual(encoder.command, "encoder-stack")
        self.assertEqual(encoder.mode, "cold")
        decoder = parser.parse_args(
            [
                "decoder-layer",
                "--weights",
                "model.gacweights",
                "--layer",
                "5",
                "-o",
                "decoder.hxp",
            ]
        )
        self.assertEqual(decoder.layer, 5)
        detection = parser.parse_args(
            [
                "detection-head",
                "--weights",
                "model.gacweights",
                "--layer",
                "0",
                "-o",
                "head.hxp",
            ]
        )
        self.assertEqual(detection.command, "detection-head")
        for command in (
            "feature-prep",
            "encoder-inputs",
            "decoder-stack",
            "detection-stack",
            "decoder-pipeline",
            "cold-pipeline",
            "history-encoder",
            "step-pipeline",
        ):
            parsed = parser.parse_args(
                [
                    command,
                    "--weights",
                    "model.gacweights",
                    "-o",
                    f"{command}.hxp",
                ]
            )
            self.assertEqual(parsed.command, command)

    def test_history_encoder_cli_dispatches_safe_archive_builder(self):
        expected = {"artifact": "hxp", "component": "history.image_to_bev_encoder"}
        output = io.StringIO()
        with patch.object(
            build_bevformer_tool,
            "build_history_encoder_package",
            return_value=expected,
        ) as builder, redirect_stdout(output):
            return_code = build_bevformer_tool.main(
                [
                    "history-encoder",
                    "--weights",
                    "model.gacweights",
                    "--expected-checkpoint-sha256",
                    CHECKPOINT_SHA256,
                    "-o",
                    "history.hxp",
                ]
            )
        self.assertEqual(return_code, 0)
        self.assertEqual(json.loads(output.getvalue()), expected)
        builder.assert_called_once_with(
            Path("model.gacweights"),
            Path("history.hxp"),
            expected_checkpoint_sha256=CHECKPOINT_SHA256,
        )

    def test_streaming_step_cli_dispatches_safe_archive_builder(self):
        expected = {
            "artifact": "hxp",
            "component": "streaming.complete_cold_and_warm_step",
        }
        output = io.StringIO()
        with patch.object(
            build_bevformer_tool,
            "build_streaming_step_package",
            return_value=expected,
        ) as builder, redirect_stdout(output):
            return_code = build_bevformer_tool.main(
                [
                    "step-pipeline",
                    "--weights",
                    "model.gacweights",
                    "--expected-checkpoint-sha256",
                    CHECKPOINT_SHA256,
                    "-o",
                    "step.hxp",
                ]
            )
        self.assertEqual(return_code, 0)
        self.assertEqual(json.loads(output.getvalue()), expected)
        builder.assert_called_once_with(
            Path("model.gacweights"),
            Path("step.hxp"),
            expected_checkpoint_sha256=CHECKPOINT_SHA256,
        )

    def test_official_decoder_and_head_names_are_transposed_into_canonical_layout(self):
        (
            state,
            _,
            _,
            decoder_cross,
            _,
            decoder_self,
            refinement,
            _,
            detection,
            feature_prep,
            encoder_inputs,
        ) = make_attention_weights()
        decoder_layer = BevFormerDecoderLayerConfig(
            self_attention=decoder_self,
            cross_attention=decoder_cross,
            ffn=BEVFormerFFNConfig(bev_h=1, bev_w=3, embed_dim=4),
        )
        ffn_stored = np.arange(32, dtype=np.float32).reshape(8, 4)
        state[
            "pts_bbox_head.transformer.decoder.layers.0.ffns.0.layers.0.0.weight"
        ] = ffn_stored
        cls_stored = np.arange(8, dtype=np.float32).reshape(2, 4)
        state["pts_bbox_head.cls_branches.0.6.weight"] = cls_stored
        reg_stored = np.arange(24, dtype=np.float32).reshape(6, 4)
        state["pts_bbox_head.reg_branches.0.4.weight"] = reg_stored

        decoder_groups = extract_decoder_layer_weights(
            state, 0, decoder_layer
        )
        np.testing.assert_array_equal(
            decoder_groups["layer"]["ffn_fc1_weight"],
            ffn_stored.T,
        )
        classification = extract_detection_head_weights(state, 0, detection)
        np.testing.assert_array_equal(classification["cls2_weight"], cls_stored.T)
        regression = extract_decoder_refinement_weights(state, 0, refinement)
        np.testing.assert_array_equal(regression["fc2_weight"], reg_stored.T)
        prepared = extract_feature_prep_weights(state, feature_prep)
        np.testing.assert_array_equal(
            prepared["camera_embeddings"],
            state["pts_bbox_head.transformer.cams_embeds"],
        )
        np.testing.assert_array_equal(
            prepared["level_embedding"],
            state["pts_bbox_head.transformer.level_embeds"],
        )
        can_bus_stored = np.arange(15, dtype=np.float32).reshape(5, 3)
        state["pts_bbox_head.transformer.can_bus_mlp.0.weight"] = can_bus_stored
        prepared_inputs = extract_encoder_input_weights(state, encoder_inputs)
        np.testing.assert_array_equal(
            prepared_inputs["can_bus_fc0_weight"],
            can_bus_stored.T,
        )
        np.testing.assert_array_equal(
            prepared_inputs["bev_embedding"],
            state["pts_bbox_head.bev_embedding.weight"],
        )
        np.testing.assert_array_equal(
            prepared_inputs["row_embedding"],
            state["pts_bbox_head.positional_encoding.row_embed.weight"],
        )
        np.testing.assert_array_equal(
            prepared_inputs["col_embedding"],
            state["pts_bbox_head.positional_encoding.col_embed.weight"],
        )
        decoder_stack = extract_decoder_stack_weights(
            state,
            BevFormerDecoderStackConfig(
                query_init=DecoderQueryInitConfig(queries=3, embed_dim=4),
                layer=decoder_layer,
                refinement=refinement,
            ),
        )
        self.assertEqual(len(decoder_stack["self_attention"]), 6)
        self.assertEqual(len(decoder_stack["cross_attention"]), 6)
        self.assertEqual(len(decoder_stack["layer"]), 6)
        self.assertEqual(len(decoder_stack["refinement"]), 6)
        detection_stack = extract_detection_stack_weights(
            state,
            DetectionStackConfig(head=detection),
        )
        self.assertEqual(len(detection_stack), 6)

    def test_official_four_row_level_embedding_maps_to_tiny_single_level(self):
        config = BevFormerFeaturePrepConfig()
        official_levels = np.arange(4 * 256, dtype=np.float32).reshape(4, 256)
        state = {
            "pts_bbox_head.transformer.cams_embeds": np.zeros(
                (6, 256), dtype=np.float32
            ),
            "pts_bbox_head.transformer.level_embeds": official_levels,
        }

        prepared = extract_feature_prep_weights(state, config)

        self.assertEqual(prepared["level_embedding"].shape, (1, 256))
        np.testing.assert_array_equal(
            prepared["level_embedding"], official_levels[:1]
        )

    def test_level_embedding_prefix_mapping_rejects_other_profiles_or_dimensions(self):
        config = BevFormerFeaturePrepConfig()
        cameras = np.zeros((6, 256), dtype=np.float32)
        invalid_levels = (
            np.empty((0, 256), dtype=np.float32),
            np.zeros((2, 256), dtype=np.float32),
            np.zeros((3, 256), dtype=np.float32),
            np.zeros((5, 256), dtype=np.float32),
            np.zeros((4, 255), dtype=np.float32),
            np.zeros((4, 1, 256), dtype=np.float32),
        )
        for value in invalid_levels:
            with self.subTest(shape=value.shape), self.assertRaisesRegex(
                ValidationError, "requires leading row count 1 or 4"
            ):
                extract_feature_prep_weights(
                    {
                        "pts_bbox_head.transformer.cams_embeds": cameras,
                        "pts_bbox_head.transformer.level_embeds": value,
                    },
                    config,
                )

    def test_official_level_embedding_validates_all_source_rows_before_slicing(self):
        config = BevFormerFeaturePrepConfig()
        cameras = np.zeros((6, 256), dtype=np.float32)
        non_finite = np.zeros((4, 256), dtype=np.float32)
        non_finite[3, 0] = np.nan
        non_contiguous = np.zeros((4, 512), dtype=np.float32)[:, ::2]
        invalid_levels = (
            ("float64", np.zeros((4, 256), dtype=np.float64), "float32"),
            ("unused-row-nan", non_finite, "contiguous and finite"),
            ("non-contiguous", non_contiguous, "contiguous and finite"),
        )
        for label, value, message in invalid_levels:
            with self.subTest(case=label), self.assertRaisesRegex(
                ValidationError, message
            ):
                extract_feature_prep_weights(
                    {
                        "pts_bbox_head.transformer.cams_embeds": cameras,
                        "pts_bbox_head.transformer.level_embeds": value,
                    },
                    config,
                )

    def test_attention_stage_packages_map_official_weights_and_roundtrip(self):
        (
            state,
            temporal_config,
            spatial_config,
            decoder_config,
            ffn_config,
            decoder_self_config,
            refinement_config,
            query_init_config,
            detection_config,
            feature_prep_config,
            encoder_inputs_config,
        ) = make_attention_weights()
        weights = write_weights(
            self.root / "attention.gacweights",
            state,
            source_checkpoint_sha256=CHECKPOINT_SHA256,
        )
        temporal = build_temporal_attention_package(
            weights,
            self.root / "temporal.hxp",
            layer=0,
            config=temporal_config,
        )
        spatial = build_spatial_attention_package(
            weights,
            self.root / "spatial.hxp",
            layer=0,
            config=spatial_config,
        )
        decoder = build_decoder_cross_attention_package(
            weights,
            self.root / "decoder.hxp",
            layer=0,
            config=decoder_config,
        )
        ffn = build_encoder_ffn_package(
            weights,
            self.root / "ffn.hxp",
            layer=0,
            config=ffn_config,
        )
        decoder_self = build_decoder_self_attention_package(
            weights,
            self.root / "decoder-self.hxp",
            layer=0,
            config=decoder_self_config,
        )
        refinement = build_decoder_refinement_package(
            weights,
            self.root / "decoder-refinement.hxp",
            layer=0,
            config=refinement_config,
        )
        encoder_layer_config = BevFormerEncoderLayerConfig(
            temporal=temporal_config,
            spatial=spatial_config,
            projection=BevProjectionConfig(
                cameras=2,
                bev_h=2,
                bev_w=2,
                z_anchors=2,
                image_h=20,
                image_w=30,
            ),
            ffn_hidden_dim=8,
        )
        encoder_layer = build_encoder_layer_package(
            weights,
            self.root / "encoder-layer.hxp",
            layer=0,
            config=encoder_layer_config,
        )
        encoder_stack_config = BevFormerEncoderStackConfig(
            layer=encoder_layer_config,
        )
        encoder_stack = build_encoder_stack_package(
            weights,
            self.root / "encoder-stack.hxp",
            config=encoder_stack_config,
        )
        decoder_layer_config = BevFormerDecoderLayerConfig(
            self_attention=decoder_self_config,
            cross_attention=decoder_config,
            ffn=BEVFormerFFNConfig(bev_h=1, bev_w=3, embed_dim=4),
        )
        decoder_layer = build_decoder_layer_package(
            weights,
            self.root / "decoder-layer.hxp",
            layer=0,
            config=decoder_layer_config,
        )
        decoder_stack_config = BevFormerDecoderStackConfig(
            query_init=query_init_config,
            layer=decoder_layer_config,
            refinement=refinement_config,
        )
        decoder_stack = build_decoder_stack_package(
            weights,
            self.root / "decoder-stack.hxp",
            config=decoder_stack_config,
        )
        query_init = build_decoder_query_init_package(
            weights,
            self.root / "decoder-query-init.hxp",
            config=query_init_config,
        )
        detection = build_detection_head_package(
            weights,
            self.root / "detection-head.hxp",
            layer=0,
            config=detection_config,
        )
        detection_stack_config = DetectionStackConfig(head=detection_config)
        detection_stack = build_detection_stack_package(
            weights,
            self.root / "detection-stack.hxp",
            config=detection_stack_config,
        )
        decoder_pipeline = build_decoder_pipeline_package(
            weights,
            self.root / "decoder-pipeline.hxp",
            config=BevFormerDecoderPipelineConfig(
                decoder=decoder_stack_config,
                detection=detection_stack_config,
            ),
        )
        feature_prep = build_feature_prep_package(
            weights,
            self.root / "feature-prep.hxp",
            config=feature_prep_config,
        )
        encoder_inputs = build_encoder_inputs_package(
            weights,
            self.root / "encoder-inputs.hxp",
            config=encoder_inputs_config,
        )
        for report, component in (
            (temporal, "encoder.temporal_self_attention"),
            (spatial, "encoder.spatial_cross_attention"),
            (decoder, "decoder.custom_msda_cross_attention"),
            (ffn, "encoder.ffn_and_norm2"),
            (decoder_self, "decoder.self_attention_and_norm0"),
            (refinement, "decoder.regression_reference_refinement"),
            (encoder_layer, "encoder.complete_layer"),
            (decoder_layer, "decoder.complete_layer"),
            (detection, "decoder.raw_detection_head"),
        ):
            self.assertEqual(report["component"], component)
            self.assertEqual(report["layer_index"], 0)
            self.assertEqual(report["model"], "BEVFormer-custom-static")
            self.assertTrue(report["codegen_verified"])
            self.assertFalse(report["physical_cuda_execution"])
            loaded = load_package(report["output"])
            self.assertEqual(loaded.manifest["metadata"]["component"], component)
        self.assertEqual(encoder_stack["component"], "encoder.complete_stack")
        self.assertNotIn("layer_index", encoder_stack)
        self.assertEqual(encoder_stack["model"], "BEVFormer-custom-static")
        self.assertTrue(encoder_stack["codegen_verified"])
        stack_package = load_package(encoder_stack["output"])
        self.assertTrue(stack_package.graph.metadata["full_encoder_stack"])
        self.assertEqual(stack_package.graph.metadata["num_layers"], 3)
        self.assertEqual(decoder_stack["component"], "decoder.complete_stack")
        self.assertNotIn("layer_index", decoder_stack)
        self.assertEqual(decoder_stack["model"], "BEVFormer-custom-static")
        self.assertTrue(decoder_stack["codegen_verified"])
        self.assertTrue(decoder_stack["full_decoder_stack"])
        self.assertEqual(decoder_stack["num_layers"], 6)
        decoder_stack_package = load_package(decoder_stack["output"])
        self.assertTrue(
            decoder_stack_package.manifest["metadata"]["full_decoder_stack"]
        )
        self.assertTrue(
            decoder_stack_package.graph.metadata["full_decoder_stack"]
        )
        self.assertEqual(
            query_init["component"],
            "decoder.query_and_reference_initialization",
        )
        self.assertNotIn("layer_index", query_init)
        self.assertEqual(query_init["model"], "BEVFormer-custom-static")
        self.assertTrue(query_init["codegen_verified"])
        self.assertEqual(feature_prep["component"], "encoder.feature_preparation")
        self.assertNotIn("layer_index", feature_prep)
        self.assertEqual(feature_prep["model"], "BEVFormer-custom-static")
        self.assertTrue(feature_prep["codegen_verified"])
        feature_graph = load_package(feature_prep["output"]).graph
        self.assertEqual(feature_graph.outputs, ("image_value",))
        self.assertEqual(encoder_inputs["component"], "encoder.input_preparation")
        self.assertNotIn("layer_index", encoder_inputs)
        self.assertEqual(encoder_inputs["model"], "BEVFormer-custom-static")
        self.assertTrue(encoder_inputs["codegen_verified"])
        encoder_inputs_graph = load_package(encoder_inputs["output"]).graph
        self.assertEqual(
            encoder_inputs_graph.outputs,
            ("query", "query_pos", "spatial_query_pos", "reference_points"),
        )
        detection_graph = load_package(detection["output"]).graph
        self.assertEqual(
            detection_graph.metadata["nms_free_topk"],
            "outside-this-graph",
        )
        self.assertEqual(
            detection_stack["component"],
            "decoder.detection_projection_stack",
        )
        self.assertNotIn("layer_index", detection_stack)
        self.assertTrue(detection_stack["full_detection_projection_stack"])
        self.assertEqual(
            detection_stack["nms_free_topk"],
            "outside-this-graph",
        )
        detection_stack_package = load_package(detection_stack["output"])
        self.assertTrue(
            detection_stack_package.graph.metadata[
                "full_detection_projection_stack"
            ]
        )
        self.assertEqual(
            decoder_pipeline["component"],
            "decoder.complete_raw_detection_pipeline",
        )
        self.assertNotIn("layer_index", decoder_pipeline)
        self.assertTrue(decoder_pipeline["full_decoder_pipeline"])
        self.assertEqual(
            decoder_pipeline["host_postprocess"],
            "decode_nms_free",
        )
        decoder_pipeline_package = load_package(decoder_pipeline["output"])
        self.assertTrue(
            decoder_pipeline_package.graph.metadata["decoder_to_raw_head_connected"]
        )

    def test_history_encoder_package_roundtrip_records_external_motion_boundary(self):
        (
            state,
            temporal_config,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            encoder_inputs_config,
        ) = make_attention_weights()
        backbone = ResNetFpnProfile(
            name="bevformer-history-builder-small",
            batch=2,
            input_channels=3,
            input_hw=(8, 8),
            stem_channels=2,
            stage_planes=(2, 3, 4, 5),
            blocks=(1, 1, 1, 1),
            expansion=2,
            fpn_channels=4,
            input_dtype="float32",
        )
        for spec in required_checkpoint_specs(backbone):
            state[spec.key] = (
                np.ones(spec.shape, dtype=np.float32)
                if spec.role == "bn-running-variance"
                else np.zeros(spec.shape, dtype=np.float32)
            )
        spatial = SpatialCrossAttentionConfig(
            cameras=2,
            queries=4,
            embed_dim=4,
            heads=2,
            spatial_shapes=((1, 1),),
            sampling_points=4,
            z_anchors=2,
        )
        encoder_layer = BevFormerEncoderLayerConfig(
            temporal=temporal_config,
            spatial=spatial,
            projection=BevProjectionConfig(
                cameras=2,
                bev_h=2,
                bev_w=2,
                z_anchors=2,
                image_h=8,
                image_w=8,
            ),
            ffn_hidden_dim=8,
        )
        config = BevFormerHistoryPipelineConfig(
            backbone=backbone,
            feature_prep=BevFormerFeaturePrepConfig(
                cameras=2,
                embed_dim=4,
                feature_h=1,
                feature_w=1,
            ),
            encoder_inputs=encoder_inputs_config,
            encoder_stack=BevFormerEncoderStackConfig(
                layer=encoder_layer,
                history_mode=True,
            ),
        )
        weights = write_weights(
            self.root / "history.gacweights",
            state,
            source_checkpoint_sha256=CHECKPOINT_SHA256,
        )
        report = build_history_encoder_package(
            weights,
            self.root / "history-encoder.hxp",
            expected_checkpoint_sha256=CHECKPOINT_SHA256,
            config=config,
        )
        self.assertEqual(report["component"], "history.image_to_bev_encoder")
        self.assertEqual(report["model"], "BEVFormer-custom-warm-history-static")
        self.assertTrue(report["warm_history_encoder"])
        self.assertEqual(
            report["motion_compensation_boundary"],
            "external-before-this-graph",
        )
        self.assertEqual(
            report["required_aligned_history_inputs"],
            ["aligned_prev_bev", "prev_reference_points"],
        )
        self.assertTrue(report["codegen_verified"])
        loaded = load_package(report["output"])
        metadata = loaded.manifest["metadata"]
        self.assertEqual(
            metadata["motion_compensation_boundary"],
            "external-before-this-graph",
        )
        self.assertEqual(
            set(loaded.graph.inputs),
            {
                "images",
                "lidar2img",
                "can_bus_delta",
                "aligned_prev_bev",
                "prev_reference_points",
            },
        )
        self.assertEqual(loaded.graph.outputs, ("next_bev",))
        self.assertTrue(loaded.graph.metadata["history_mode"])
        self.assertIn("external", loaded.graph.metadata["history_alignment"])

    def test_cold_single_frame_pipeline_package_roundtrip_and_boundaries(self):
        (
            state,
            temporal_config,
            _,
            decoder_cross_config,
            _,
            decoder_self_config,
            refinement_config,
            query_init_config,
            detection_config,
            _,
            encoder_inputs_config,
        ) = make_attention_weights()
        backbone = ResNetFpnProfile(
            name="bevformer-cold-builder-small",
            batch=2,
            input_channels=3,
            input_hw=(8, 8),
            stem_channels=2,
            stage_planes=(2, 3, 4, 5),
            blocks=(1, 1, 1, 1),
            expansion=2,
            fpn_channels=4,
            input_dtype="float32",
        )
        for spec in required_checkpoint_specs(backbone):
            state[spec.key] = (
                np.ones(spec.shape, dtype=np.float32)
                if spec.role == "bn-running-variance"
                else np.zeros(spec.shape, dtype=np.float32)
            )
        spatial = SpatialCrossAttentionConfig(
            cameras=2,
            queries=4,
            embed_dim=4,
            heads=2,
            spatial_shapes=((1, 1),),
            sampling_points=4,
            z_anchors=2,
        )
        encoder_layer = BevFormerEncoderLayerConfig(
            temporal=temporal_config,
            spatial=spatial,
            projection=BevProjectionConfig(
                cameras=2,
                bev_h=2,
                bev_w=2,
                z_anchors=2,
                image_h=8,
                image_w=8,
            ),
            ffn_hidden_dim=8,
        )
        encoder = BevFormerColdEncoderPipelineConfig(
            backbone=backbone,
            feature_prep=BevFormerFeaturePrepConfig(
                cameras=2,
                embed_dim=4,
                feature_h=1,
                feature_w=1,
            ),
            encoder_inputs=encoder_inputs_config,
            encoder_stack=BevFormerEncoderStackConfig(
                layer=encoder_layer,
                history_mode=False,
            ),
        )
        decoder_layer = BevFormerDecoderLayerConfig(
            self_attention=decoder_self_config,
            cross_attention=decoder_cross_config,
            ffn=BEVFormerFFNConfig(bev_h=1, bev_w=3, embed_dim=4),
        )
        decoder = BevFormerDecoderPipelineConfig(
            decoder=BevFormerDecoderStackConfig(
                query_init=query_init_config,
                layer=decoder_layer,
                refinement=refinement_config,
            ),
            detection=DetectionStackConfig(head=detection_config),
        )
        config = BevFormerColdPipelineConfig(encoder=encoder, decoder=decoder)
        weights = write_weights(
            self.root / "cold.gacweights",
            state,
            source_checkpoint_sha256=CHECKPOINT_SHA256,
        )
        report = build_cold_pipeline_package(
            weights,
            self.root / "cold-pipeline.hxp",
            config=config,
        )
        self.assertEqual(
            report["component"],
            "cold.single_frame_image_to_raw_detections",
        )
        self.assertEqual(report["model"], "BEVFormer-custom-cold-static")
        self.assertNotIn("layer_index", report)
        self.assertTrue(report["cold_single_frame_only"])
        self.assertFalse(report["streaming_step"])
        self.assertEqual(report["host_postprocess"], "decode_nms_free")
        loaded = load_package(report["output"])
        metadata = loaded.manifest["metadata"]
        self.assertTrue(metadata["cold_single_frame_only"])
        self.assertFalse(metadata["streaming_step"])
        self.assertEqual(metadata["host_postprocess"], "decode_nms_free")
        self.assertTrue(loaded.graph.metadata["cold_single_frame_core_connected"])
        self.assertTrue(loaded.graph.metadata["not_a_streaming_step"])

    def test_streaming_step_package_roundtrip_has_exact_stateful_abi(self):
        (
            state,
            temporal_config,
            _,
            decoder_cross_config,
            _,
            decoder_self_config,
            _,
            query_init_config,
            _,
            _,
            _,
        ) = make_attention_weights()
        backbone = ResNetFpnProfile(
            name="bevformer-step-builder-small",
            batch=2,
            input_channels=3,
            input_hw=(8, 8),
            stem_channels=2,
            stage_planes=(2, 3, 4, 5),
            blocks=(1, 1, 1, 1),
            expansion=2,
            fpn_channels=4,
            input_dtype="float16",
        )
        for spec in required_checkpoint_specs(backbone):
            state[spec.key] = (
                np.ones(spec.shape, dtype=np.float32)
                if spec.role == "bn-running-variance"
                else np.zeros(spec.shape, dtype=np.float32)
            )

        # The public step follows the official 18-value CAN-bus contract and
        # ten-channel NMS-free box representation.  Replace only those final
        # official checkpoint tensors in the small fixture.
        add_linear(
            state,
            "pts_bbox_head.transformer.can_bus_mlp.0",
            18,
            5,
        )
        for layer_index in range(6):
            add_linear(
                state,
                f"pts_bbox_head.cls_branches.{layer_index}.6",
                4,
                10,
            )
            add_linear(
                state,
                f"pts_bbox_head.reg_branches.{layer_index}.4",
                4,
                10,
            )

        point_cloud_range = (-2.0, -3.0, -1.0, 2.0, 5.0, 3.0)
        spatial = SpatialCrossAttentionConfig(
            cameras=2,
            queries=4,
            embed_dim=4,
            heads=2,
            spatial_shapes=((1, 1),),
            sampling_points=4,
            z_anchors=2,
        )
        encoder_layer = BevFormerEncoderLayerConfig(
            temporal=temporal_config,
            spatial=spatial,
            projection=BevProjectionConfig(
                cameras=2,
                bev_h=2,
                bev_w=2,
                z_anchors=2,
                image_h=8,
                image_w=8,
                point_cloud_range=point_cloud_range,
            ),
            ffn_hidden_dim=8,
        )
        encoder = BevFormerHistoryPipelineConfig(
            backbone=backbone,
            feature_prep=BevFormerFeaturePrepConfig(
                cameras=2,
                embed_dim=4,
                feature_h=1,
                feature_w=1,
            ),
            encoder_inputs=BevFormerEncoderInputConfig(
                bev_h=2,
                bev_w=2,
                embed_dim=4,
                can_bus_dim=18,
                can_bus_hidden_dim=5,
            ),
            encoder_stack=BevFormerEncoderStackConfig(
                layer=encoder_layer,
                history_mode=True,
            ),
        )
        decoder_layer = BevFormerDecoderLayerConfig(
            self_attention=decoder_self_config,
            cross_attention=decoder_cross_config,
            ffn=BEVFormerFFNConfig(bev_h=1, bev_w=3, embed_dim=4),
        )
        decoder = BevFormerDecoderPipelineConfig(
            decoder=BevFormerDecoderStackConfig(
                query_init=query_init_config,
                layer=decoder_layer,
                refinement=DecoderReferenceRefinementConfig(
                    queries=3,
                    embed_dim=4,
                    code_size=10,
                ),
            ),
            detection=DetectionStackConfig(
                head=DetectionHeadConfig(
                    queries=3,
                    embed_dim=4,
                    classes=10,
                    code_size=10,
                    point_cloud_range=point_cloud_range,
                )
            ),
        )
        config = BevFormerStepPipelineConfig(
            encoder=encoder,
            motion=BevFormerMotionConfig(
                bev_h=2,
                bev_w=2,
                embed_dim=4,
                grid_length_y=4.0,
                grid_length_x=2.0,
                rotate_center=(1.0, 1.0),
            ),
            decoder=decoder,
        )
        weights = write_weights(
            self.root / "step.gacweights",
            state,
            source_checkpoint_sha256=CHECKPOINT_SHA256,
        )
        report = build_streaming_step_package(
            weights,
            self.root / "step-pipeline.hxp",
            expected_checkpoint_sha256=CHECKPOINT_SHA256,
            config=config,
        )

        self.assertEqual(
            report["component"],
            "streaming.complete_cold_and_warm_step",
        )
        self.assertEqual(report["model"], "BEVFormer-custom-streaming-step-static")
        self.assertTrue(report["streaming_step"])
        self.assertTrue(report["cold_and_warm"])
        self.assertEqual(report["host_postprocess"], "decode_nms_free")
        self.assertEqual(report["topk_boundary"], "host-postprocess")
        self.assertEqual(report["num_encoder_layers"], 3)
        self.assertEqual(report["num_decoder_layers"], 6)
        self.assertEqual(report["num_detection_layers"], 6)
        self.assertEqual(
            report["raw_box_representation"],
            "[physical-x,physical-y,log-w,log-l,physical-z,log-h,"
            "sin-yaw,cos-yaw,vx,vy]",
        )
        self.assertTrue(report["codegen_verified"])

        loaded = load_package(report["output"])
        # Package JSON canonicalizes mapping keys; names and tensor contracts
        # are the ABI, while output tuple order remains semantically ordered.
        self.assertEqual(
            set(loaded.graph.inputs),
            {
                "images",
                "lidar2img",
                "can_bus_delta",
                "prev_bev",
                "has_prev_bev",
            },
        )
        self.assertEqual(
            loaded.graph.outputs,
            ("cls_logits", "bbox_regression", "next_bev"),
        )
        expected_inputs = {
            "images": ((1, 2, 3, 8, 8), "float16"),
            "lidar2img": ((1, 2, 4, 4), "float32"),
            "can_bus_delta": ((1, 18), "float32"),
            "prev_bev": ((1, 4, 4), "float16"),
            "has_prev_bev": ((1,), "int32"),
        }
        self.assertEqual(
            {
                name: (spec.shape, spec.dtype)
                for name, spec in loaded.graph.inputs.items()
            },
            expected_inputs,
        )
        self.assertEqual(
            {
                name: (
                    loaded.graph.value_specs[name].shape,
                    loaded.graph.value_specs[name].dtype,
                )
                for name in loaded.graph.outputs
            },
            {
                "cls_logits": ((1, 3, 10), "float32"),
                "bbox_regression": ((1, 3, 10), "float32"),
                "next_bev": ((1, 4, 4), "float16"),
            },
        )
        metadata = loaded.manifest["metadata"]
        self.assertTrue(metadata["streaming_step"])
        self.assertTrue(metadata["cold_and_warm"])
        self.assertEqual(metadata["host_postprocess"], "decode_nms_free")
        self.assertEqual(metadata["topk_boundary"], "host-postprocess")


if __name__ == "__main__":
    unittest.main()
