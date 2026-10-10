"""Offline tests for the BEVFormer tensor ABI and safe config reader."""

from __future__ import annotations

import json
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

import numpy as np

from gac.frontends.bevformer_export import (
    ENTRYPOINT,
    MANIFEST_FORMAT,
    NUSCENES_CAMERA_ORDER,
    PROFILE_NAME,
    BevFormerExportError,
    BevFormerStepManifest,
    audit_python_source,
    build_bevformer_tiny_manifest,
    extract_bevformer_tiny_config,
    load_bevformer_manifest,
    save_bevformer_manifest,
)


ROOT = Path(__file__).resolve().parents[1]
LOCAL_CONFIG = Path(
    r"C:\Users\Thomas\Desktop\project\BEVFormer-master"
) / "projects/configs/bevformer/bevformer_tiny.py"


CONFIG_SOURCE = textwrap.dedent(
    """
    _base_ = ['../datasets/custom_nus-3d.py', '../_base_/default_runtime.py']
    plugin = True
    plugin_dir = 'projects/mmdet3d_plugin/'
    point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
    voxel_size = [0.2, 0.2, 8]
    img_norm_cfg = dict(
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True)
    class_names = [
        'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
        'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']
    input_modality = dict(
        use_lidar=False, use_camera=True, use_radar=False, use_map=False,
        use_external=True)
    _dim_ = 256
    _pos_dim_ = _dim_ // 2
    _ffn_dim_ = _dim_ * 2
    _num_levels_ = 1
    bev_h_ = 50
    bev_w_ = 50
    queue_length = 3
    model = dict(
        type='BEVFormer',
        img_backbone=dict(type='ResNet', depth=50, out_indices=(3,)),
        img_neck=dict(type='FPN', in_channels=[2048], out_channels=_dim_,
                      num_outs=_num_levels_),
        pts_bbox_head=dict(
            type='BEVFormerHead', bev_h=bev_h_, bev_w=bev_w_, num_query=900,
            num_classes=10, in_channels=_dim_,
            transformer=dict(
                type='PerceptionTransformer', embed_dims=_dim_,
                encoder=dict(
                    type='BEVFormerEncoder', num_layers=3,
                    transformerlayers=dict(
                        type='BEVFormerLayer',
                        attn_cfgs=[
                            dict(type='TemporalSelfAttention', embed_dims=_dim_,
                                 num_levels=1),
                            dict(
                                type='SpatialCrossAttention',
                                deformable_attention=dict(
                                    type='MSDeformableAttention3D',
                                    embed_dims=_dim_, num_points=8,
                                    num_levels=_num_levels_))])),
                decoder=dict(
                    type='DetectionTransformerDecoder', num_layers=6,
                    transformerlayers=dict(
                        type='DetrTransformerDecoderLayer',
                        attn_cfgs=[
                            dict(type='MultiheadAttention', embed_dims=_dim_,
                                 num_heads=8),
                            dict(type='CustomMSDeformableAttention',
                                 embed_dims=_dim_, num_levels=1)]))),
            positional_encoding=dict(
                type='LearnedPositionalEncoding', num_feats=_pos_dim_,
                row_num_embed=bev_h_, col_num_embed=bev_w_),
            bbox_coder=dict(type='NMSFreeCoder', num_classes=10)))
    test_pipeline = [
        dict(type='LoadMultiViewImageFromFiles', to_float32=True),
        dict(type='NormalizeMultiviewImage', **img_norm_cfg),
        dict(
            type='MultiScaleFlipAug3D', img_scale=(1600, 900),
            transforms=[
                dict(type='RandomScaleImageMultiViewImage', scales=[0.5]),
                dict(type='PadMultiViewImage', size_divisor=32)])]
    data = dict(samples_per_gpu=1)
    """
).lstrip()


class ConfigReaderTests(unittest.TestCase):
    def _write_config(self, source: str = CONFIG_SOURCE) -> Path:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "bevformer_tiny.py"
        path.write_text(source, encoding="utf-8")
        return path

    def test_extracts_frozen_tiny_profile_without_framework_imports(self):
        before = {name for name in sys.modules if name.startswith(("torch", "mmcv"))}
        extraction = extract_bevformer_tiny_config(self._write_config())
        after = {name for name in sys.modules if name.startswith(("torch", "mmcv"))}

        self.assertEqual(after, before)
        self.assertTrue(extraction.audit.safe_for_static_config)
        self.assertEqual(extraction.config.image_content_hw, (450, 800))
        self.assertEqual(extraction.config.image_input_hw, (480, 800))
        self.assertEqual(extraction.config.feature_shapes, ((15, 25),))
        self.assertEqual(extraction.config.bev_queries, 2500)
        self.assertEqual(extraction.config.num_classes, 10)
        self.assertEqual(extraction.config.embed_dim, 256)
        self.assertEqual(extraction.config.encoder_layers, 3)
        self.assertEqual(extraction.config.decoder_layers, 6)
        self.assertEqual(extraction.config.sampling_points, 8)
        self.assertEqual(len(extraction.source_sha256), 64)

    def test_real_local_config_matches_when_available(self):
        if not LOCAL_CONFIG.exists():
            self.skipTest("local BEVFormer snapshot is not present")
        extraction = extract_bevformer_tiny_config(LOCAL_CONFIG)
        self.assertEqual(
            extraction.source_sha256,
            "85da2b393cb0103645bdc034c87028b6fcf5d7bdd79530ed6b4f4b2d05777882",
        )
        self.assertEqual(extraction.config.image_input_hw, (480, 800))

    def test_audit_reports_import_call_and_control_flow(self):
        report = audit_python_source(
            "import os\nx = os.system('must-not-run')\nif x:\n    y = 1\n",
            source_name="hostile.py",
        )
        self.assertFalse(report.safe_for_static_config)
        self.assertEqual(report.source, "hostile.py")
        codes = {finding.code for finding in report.findings}
        self.assertIn("UNSAFE_IMPORT", codes)
        self.assertIn("UNSAFE_CALL", codes)
        self.assertIn("DYNAMIC_STATEMENT", codes)
        self.assertTrue(all(finding.line >= 1 for finding in report.findings))

    def test_audit_reports_unresolved_name_and_syntax_error(self):
        unresolved = audit_python_source("x = missing + 1\n")
        self.assertEqual(unresolved.findings[0].code, "UNRESOLVED_NAME")
        syntax = audit_python_source("x = [\n")
        self.assertEqual(syntax.findings[0].code, "SYNTAX_ERROR")

    def test_extractor_exposes_audit_on_rejection(self):
        path = self._write_config("import pathlib\nmodel = pathlib.Path('.')\n")
        with self.assertRaises(BevFormerExportError) as caught:
            extract_bevformer_tiny_config(path)
        self.assertIsNotNone(caught.exception.audit_report)
        self.assertFalse(caught.exception.audit_report.safe_for_static_config)

    def test_profile_change_is_rejected(self):
        changed = CONFIG_SOURCE.replace("_dim_ = 256", "_dim_ = 128")
        with self.assertRaisesRegex(BevFormerExportError, "frozen BEVFormer-tiny"):
            extract_bevformer_tiny_config(self._write_config(changed))

    def test_requires_official_learned_positional_encoding(self):
        changed = CONFIG_SOURCE.replace(
            "type='LearnedPositionalEncoding'",
            "type='SinePositionalEncoding'",
        )
        with self.assertRaisesRegex(
            BevFormerExportError, "LearnedPositionalEncoding"
        ):
            extract_bevformer_tiny_config(self._write_config(changed))

    def test_positional_embedding_table_shape_must_match_bev_grid(self):
        changed = CONFIG_SOURCE.replace(
            "row_num_embed=bev_h_", "row_num_embed=bev_h_ - 1"
        )
        with self.assertRaisesRegex(BevFormerExportError, "row_num_embed"):
            extract_bevformer_tiny_config(self._write_config(changed))


class ManifestTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.config_path = Path(self.directory.name) / "bevformer_tiny.py"
        self.config_path.write_text(CONFIG_SOURCE, encoding="utf-8")
        self.manifest = build_bevformer_tiny_manifest(
            self.config_path,
            source_label="projects/configs/bevformer/bevformer_tiny.py",
        )

    def test_step_abi_is_stable_and_tensor_only(self):
        manifest = self.manifest
        self.assertEqual(manifest.format, MANIFEST_FORMAT)
        self.assertEqual(manifest.profile, PROFILE_NAME)
        self.assertEqual(manifest.entrypoint, ENTRYPOINT)
        self.assertEqual(manifest.status, "offline-contract-only")
        self.assertEqual(manifest.camera_order, NUSCENES_CAMERA_ORDER)
        self.assertEqual(
            tuple(item.name for item in manifest.inputs),
            (
                "images",
                "lidar2img",
                "can_bus_delta",
                "prev_bev",
                "has_prev_bev",
            ),
        )
        self.assertEqual(
            tuple(item.shape for item in manifest.inputs),
            (
                (1, 6, 3, 480, 800),
                (1, 6, 4, 4),
                (1, 18),
                (1, 2500, 256),
                (1,),
            ),
        )
        self.assertEqual(
            tuple((item.name, item.shape) for item in manifest.outputs),
            (
                ("cls_logits", (1, 900, 10)),
                ("bbox_regression", (1, 900, 10)),
                ("next_bev", (1, 2500, 256)),
            ),
        )
        self.assertEqual(set(manifest.unavailable_exports), {"torch.export", "onnx"})

    def test_strict_json_round_trip_and_file_helpers(self):
        encoded = self.manifest.to_json()
        decoded = BevFormerStepManifest.from_json(encoded)
        self.assertEqual(decoded, self.manifest)
        self.assertEqual(decoded.to_json(), encoded)

        path = Path(self.directory.name) / "contract.json"
        save_bevformer_manifest(self.manifest, path)
        self.assertEqual(load_bevformer_manifest(path), self.manifest)
        self.assertTrue(path.read_bytes().endswith(b"\n"))

    def test_json_rejects_unknown_missing_duplicate_and_nonfinite_values(self):
        document = self.manifest.to_dict()
        document["unknown"] = 1
        with self.assertRaisesRegex(BevFormerExportError, "unexpected"):
            BevFormerStepManifest.from_dict(document)

        document = self.manifest.to_dict()
        del document["entrypoint"]
        with self.assertRaisesRegex(BevFormerExportError, "missing"):
            BevFormerStepManifest.from_dict(document)

        with self.assertRaisesRegex(BevFormerExportError, "duplicate JSON key"):
            BevFormerStepManifest.from_json('{"format": 1, "format": 2}')
        with self.assertRaisesRegex(BevFormerExportError, "numeric constant"):
            BevFormerStepManifest.from_json('{"value": NaN}')

    def _valid_inputs(self) -> dict[str, np.ndarray]:
        return {
            item.name: np.zeros(item.shape, dtype=item.dtype)
            for item in self.manifest.inputs
        }

    def _valid_outputs(self) -> dict[str, np.ndarray]:
        return {
            item.name: np.zeros(item.shape, dtype=item.dtype)
            for item in self.manifest.outputs
        }

    def test_runtime_tensor_validation(self):
        inputs = self._valid_inputs()
        self.manifest.validate_inputs(inputs)
        outputs = self._valid_outputs()
        self.manifest.validate_outputs(outputs)

        inputs["has_prev_bev"][0] = 2
        with self.assertRaisesRegex(BevFormerExportError, "only 0 or 1"):
            self.manifest.validate_inputs(inputs)

        inputs = self._valid_inputs()
        inputs["can_bus_delta"][0, 0] = 1.0
        with self.assertRaisesRegex(BevFormerExportError, "translation"):
            self.manifest.validate_inputs(inputs)

        inputs = self._valid_inputs()
        inputs["can_bus_delta"][0, -1] = 1.0
        with self.assertRaisesRegex(BevFormerExportError, "frame rotation"):
            self.manifest.validate_inputs(inputs)

        # The official first-frame path preserves absolute orientation and
        # auxiliary CAN-bus fields for the learned CAN-bus MLP.
        inputs = self._valid_inputs()
        inputs["can_bus_delta"][0, 3:-1] = np.arange(14, dtype=np.float32)
        self.manifest.validate_inputs(inputs)

        inputs = self._valid_inputs()
        inputs["has_prev_bev"][0] = 1
        inputs["can_bus_delta"][0, 0] = 1.0
        self.manifest.validate_inputs(inputs)

    def test_runtime_tensor_validation_rejects_shape_dtype_layout_and_names(self):
        cases = []
        missing = self._valid_inputs()
        del missing["images"]
        cases.append(missing)

        wrong_dtype = self._valid_inputs()
        wrong_dtype["lidar2img"] = wrong_dtype["lidar2img"].astype(np.float16)
        cases.append(wrong_dtype)

        wrong_shape = self._valid_inputs()
        wrong_shape["has_prev_bev"] = np.zeros((2,), dtype=np.int32)
        cases.append(wrong_shape)

        non_contiguous = self._valid_inputs()
        base = np.zeros((1, 6, 4, 8), dtype=np.float32)
        non_contiguous["lidar2img"] = base[..., ::2]
        self.assertFalse(non_contiguous["lidar2img"].flags.c_contiguous)
        cases.append(non_contiguous)

        non_finite = self._valid_inputs()
        non_finite["can_bus_delta"][0, 0] = np.nan
        cases.append(non_finite)

        for values in cases:
            with self.subTest(case=len(values)):
                with self.assertRaises(BevFormerExportError):
                    self.manifest.validate_inputs(values)

    def test_checked_in_example_is_strictly_loadable(self):
        example = ROOT / "examples/bevformer_tiny_contract.json"
        self.assertTrue(example.exists())
        loaded = load_bevformer_manifest(example)
        self.assertEqual(loaded.profile, PROFILE_NAME)
        self.assertEqual(loaded.status, "offline-contract-only")
        if LOCAL_CONFIG.exists():
            local = build_bevformer_tiny_manifest(
                LOCAL_CONFIG,
                source_label="projects/configs/bevformer/bevformer_tiny.py",
            )
            self.assertEqual(loaded, local)


if __name__ == "__main__":
    unittest.main()
