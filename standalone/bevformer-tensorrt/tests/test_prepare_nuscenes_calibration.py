"""Synthetic, small CPU fixtures for the reader/preprocessing contract only."""

from collections import Counter
import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from bevformer_trt.prepare_nuscenes_calibration import (
    CAMERA_ORDER, INPUT_SHAPE, MEAN, MINI_TRAIN, MINI_VAL,
    image_path, preprocess_bgr, read_mini_index, select_samples, validate_export_report,
)


def make_tables(root, samples_per_scene=8):
    tables = {name: [] for name in ("scene", "sample", "sample_data", "calibrated_sensor", "sensor")}
    for camera_index, channel in enumerate(CAMERA_ORDER):
        tables["sensor"].append({"token": f"sensor{camera_index}", "modality": "camera", "channel": channel})
        tables["calibrated_sensor"].append({"token": f"calib{camera_index}", "sensor_token": f"sensor{camera_index}"})
    for scene_index, scene in enumerate(MINI_TRAIN + MINI_VAL):
        scene_token = f"scene_token_{scene_index}"
        tables["scene"].append({"token": scene_token, "name": scene})
        for sample_index in range(samples_per_scene):
            token = f"sample_{scene_index}_{sample_index}"
            tables["sample"].append({"token": token, "scene_token": scene_token,
                                     "timestamp": 1_000_000 * scene_index + 100 * sample_index})
            for camera_index, channel in enumerate(CAMERA_ORDER):
                tables["sample_data"].append({
                    "token": f"data_{scene_index}_{sample_index}_{camera_index}", "sample_token": token,
                    "calibrated_sensor_token": f"calib{camera_index}", "is_key_frame": True,
                    "filename": f"samples/{channel}/{token}.png", "width": 3, "height": 2,
                    "timestamp": 1_000_000 * scene_index + 100 * sample_index + camera_index,
                })
    # Deliberately unrelated JSON order: the ABI must come from the sensor join.
    tables["sample_data"].reverse()
    write_tables(root, tables)
    return tables


def write_tables(root, tables):
    directory = root / "v1.0-mini"
    directory.mkdir(exist_ok=True)
    for name, records in tables.items():
        (directory / f"{name}.json").write_text(json.dumps(records), encoding="utf-8")


class PreparationTests(unittest.TestCase):
    def test_raw_tables_reconstruct_camera_order_without_sample_data_field(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tables = make_tables(root)
            self.assertNotIn("data", tables["sample"][0])
            grouped = read_mini_index(root)
            sample = grouped[MINI_TRAIN[0]][0]
            self.assertEqual(tuple(camera["channel"] for camera in sample["cameras"]), CAMERA_ORDER)
            self.assertEqual([camera["sensor_token"] for camera in sample["cameras"]],
                             [f"sensor{i}" for i in range(6)])
            self.assertEqual(sample["sample_token"], "sample_0_0")

    def test_duplicate_key_frame_rejected_nonkeyframe_ignored(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tables = make_tables(root)
            clone = dict(tables["sample_data"][0], token="another", is_key_frame=False)
            tables["sample_data"].append(clone)
            write_tables(root, tables)
            read_mini_index(root)
            clone["is_key_frame"] = True
            write_tables(root, tables)
            with self.assertRaisesRegex(ValueError, "Duplicate key-frame"):
                read_mini_index(root)

    def test_sample_data_decoration_mismatch_and_missing_camera_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tables = make_tables(root)
            tables["sample"][0]["data"] = {"CAM_FRONT": "wrong"}
            write_tables(root, tables)
            with self.assertRaisesRegex(ValueError, "disagrees"):
                read_mini_index(root)
            del tables["sample"][0]["data"]
            tables["sample_data"].pop()
            write_tables(root, tables)
            with self.assertRaisesRegex(ValueError, "missing key-frame cameras"):
                read_mini_index(root)

    def test_selection_balances_scenes_and_time_while_holding_out_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_tables(root)
            grouped = read_mini_index(root)
            selected = select_samples(grouped, 32)
            self.assertEqual(Counter(row["scene_name"] for row in selected), dict.fromkeys(MINI_TRAIN, 4))
            self.assertFalse(set(row["scene_name"] for row in selected) & set(MINI_VAL))
            self.assertEqual(len({row["sample_token"] for row in selected}), 32)
            for scene in MINI_TRAIN:
                chosen = [row["timestamp"] for row in selected if row["scene_name"] == scene]
                full = grouped[scene]
                self.assertGreater(max(chosen), full[len(full) // 2]["timestamp"])
            self.assertEqual(selected, select_samples(grouped, 32), "sampling must be deterministic")

    def test_small_sample_budget_spreads_scene_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_tables(root)
            grouped = read_mini_index(root)
            selected = select_samples(grouped, 2)
            self.assertEqual([row["scene_name"] for row in selected], [MINI_TRAIN[2], MINI_TRAIN[6]])
            with self.assertRaisesRegex(ValueError, "positive"):
                select_samples(grouped, 0)
            del grouped[MINI_VAL[0]]
            with self.assertRaisesRegex(ValueError, "Incomplete"):
                select_samples(grouped)

    def test_bgr_normalization_happens_before_zero_padding(self):
        image = np.full((2, 3, 3), [10, 20, 30], dtype=np.uint8)
        result = preprocess_bgr(image, expected_hw=(2, 3), pad_divisor=4)
        self.assertEqual(result.shape, (3, 4, 4))
        self.assertEqual(result.dtype, np.float32)
        self.assertTrue(result.flags.c_contiguous)
        np.testing.assert_allclose(result[:, 0, 0], np.array([10, 20, 30], np.float32) - MEAN, rtol=0, atol=1e-5)
        self.assertTrue((result[:, 2:, :] == 0).all())
        self.assertTrue((result[:, :, 3:] == 0).all())
        with self.assertRaisesRegex(ValueError, "no implicit resize"):
            preprocess_bgr(image)

    def test_png_decode_and_dataset_path_safety(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = np.full((2, 3, 3), [11, 22, 33], dtype=np.uint8)
            self.assertTrue(cv2.imwrite(str(root / "tiny.png"), image))
            path = image_path(root, "tiny.png")
            decoded = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
            np.testing.assert_array_equal(decoded, image)
            with self.assertRaisesRegex(ValueError, "inside"):
                image_path(root, "../outside.png")
            with self.assertRaisesRegex(ValueError, "inside"):
                image_path(root, str(path.resolve()))

    def test_export_report_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "export.json"
            report = {"input": {"images": list(INPUT_SHAPE)}, "profile": {"normalization": {
                "mean": [103.53, 116.28, 123.675], "std": [1.0, 1.0, 1.0], "to_rgb": False}}}
            path.write_text(json.dumps(report), encoding="utf-8")
            self.assertEqual(len(validate_export_report(path)["sha256"]), 64)
            report["profile"]["normalization"]["to_rgb"] = True
            path.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "disagrees"):
                validate_export_report(path)


if __name__ == "__main__":
    unittest.main()
