#!/usr/bin/env python3
"""Prepare real mini_train images for the fixed BEVFormer Base TensorRT input.

No model, GPU, dataset login or nuScenes devkit is needed. Requires NumPy and
OpenCV. mini_val stays held out. Real pixels do not correct the exported model's
fixed synthetic geometry/CAN-bus, and these files do not establish mAP or NDS.
"""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np


# Verified against BEVFormer tools/data_converter/nuscenes_converter.py and
# src/gac/models/bevformer/frontends/export.py:NUSCENES_CAMERA_ORDER. Camera
# dictionary order is not a safe substitute for this model ABI.
CAMERA_ORDER = ("CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT")
MINI_TRAIN = ("scene-0061", "scene-0553", "scene-0655", "scene-0757", "scene-0796", "scene-1077", "scene-1094", "scene-1100")
MINI_VAL = ("scene-0103", "scene-0916")
TABLES = ("scene", "sample", "sample_data", "calibrated_sensor", "sensor")
MEAN = np.array([103.53, 116.28, 123.675], dtype=np.float32)
INPUT_SHAPE = (1, 6, 3, 928, 1600)
ORIGINAL_HW = (900, 1600)
SPLIT_SOURCE = "https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/utils/splits.py"
CONFIG_SOURCE = "https://github.com/fundamentalvision/BEVFormer/blob/master/projects/configs/bevformer/bevformer_base.py"
CAMERA_SOURCE = "https://github.com/fundamentalvision/BEVFormer/blob/master/tools/data_converter/nuscenes_converter.py"
PIPELINE_SOURCE = "https://github.com/fundamentalvision/BEVFormer/blob/master/projects/mmdet3d_plugin/datasets/pipelines/transform_3d.py"


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_table(directory, name):
    records = json.loads((directory / f"{name}.json").read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError(f"{name}.json must contain a list")
    indexed = {}
    for record in records:
        token = record.get("token")
        if not isinstance(token, str) or not token or token in indexed:
            raise ValueError(f"{name}.json has an invalid/duplicate token: {token!r}")
        indexed[token] = record
    return indexed


def read_mini_index(root):
    """Join raw JSON tables; sample.data is normally added only by the devkit."""
    directory = Path(root) / "v1.0-mini"
    tables = {name: load_table(directory, name) for name in TABLES}
    cameras = defaultdict(dict)
    for data in tables["sample_data"].values():
        if not isinstance(data.get("is_key_frame"), bool):
            raise ValueError(f"sample_data {data['token']} has no boolean is_key_frame")
        if not data["is_key_frame"]:
            continue
        calibration = tables["calibrated_sensor"][data["calibrated_sensor_token"]]
        sensor = tables["sensor"][calibration["sensor_token"]]
        if sensor["modality"] != "camera":
            continue
        channel, sample_token = sensor["channel"], data["sample_token"]
        if channel not in CAMERA_ORDER:
            raise ValueError(f"Unexpected nuScenes camera: {channel}")
        if sample_token not in tables["sample"]:
            raise ValueError(f"Unknown sample_token: {sample_token}")
        if channel in cameras[sample_token]:
            raise ValueError(f"Duplicate key-frame camera {channel} for sample {sample_token}")
        cameras[sample_token][channel] = {**data, "sensor_token": sensor["token"], "channel": channel}
    grouped = defaultdict(list)
    for sample in tables["sample"].values():
        scene = tables["scene"][sample["scene_token"]]
        if scene["name"] not in MINI_TRAIN + MINI_VAL:
            raise ValueError(f"Unexpected scene in v1.0-mini: {scene['name']}")
        timestamp = sample.get("timestamp")
        if isinstance(timestamp, bool) or not isinstance(timestamp, int):
            raise ValueError(f"Sample {sample['token']} has an invalid timestamp")
        views = cameras[sample["token"]]
        if set(views) != set(CAMERA_ORDER):
            raise ValueError(f"Sample {sample['token']} is missing key-frame cameras: {sorted(set(CAMERA_ORDER) - set(views))}")
        if "data" in sample:
            for channel in CAMERA_ORDER:
                if sample["data"].get(channel) != views[channel]["token"]:
                    raise ValueError(f"sample.data disagrees with the table join: {sample['token']} {channel}")
        grouped[scene["name"]].append({
            "sample_token": sample["token"], "scene_token": scene["token"], "scene_name": scene["name"],
            "timestamp": timestamp, "cameras": [views[channel] for channel in CAMERA_ORDER],
        })
    for rows in grouped.values():
        rows.sort(key=lambda row: (row["timestamp"], row["sample_token"]))
    return dict(grouped)


def select_samples(grouped, maximum=32):
    """Balance scene quotas, then choose nearest frames to uniform time bins."""
    if maximum <= 0:
        raise ValueError("--max-samples must be positive")
    missing = [name for name in MINI_TRAIN + MINI_VAL if not grouped.get(name)]
    if missing:
        raise ValueError(f"Incomplete mini dataset; missing scenes: {missing}")
    names = list(MINI_TRAIN)
    if maximum < len(names):
        names = [names[((2 * i + 1) * len(names)) // (2 * maximum)] for i in range(maximum)]
    quotas = dict.fromkeys(names, 0)
    total = min(maximum, sum(len(grouped[name]) for name in names))
    for _ in range(total):
        eligible = [name for name in names if quotas[name] < len(grouped[name])]
        chosen = min(eligible, key=lambda name: quotas[name])
        quotas[chosen] += 1
    selected = []
    for name in names:
        rows, count = grouped[name], quotas[name]
        available = set(range(len(rows)))
        start, finish = rows[0]["timestamp"], rows[-1]["timestamp"]
        indices = []
        for i in range(count):
            target = start + (finish - start) * (i + 0.5) / count
            index = min(available, key=lambda j: (abs(rows[j]["timestamp"] - target), j))
            indices.append(index)
            available.remove(index)
        selected.extend(rows[i] for i in sorted(indices))
    return selected


def image_path(root, filename):
    root = Path(root).resolve()
    relative = Path(filename)
    path = (root / relative).resolve()
    if relative.is_absolute() or not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"Image must exist inside the dataset directory: {filename}")
    return path


def preprocess_bgr(image, *, expected_hw=ORIGINAL_HW, pad_divisor=32):
    """Match upstream NormalizeMultiviewImage followed by PadMultiViewImage.

    expected_hw/pad_divisor are explicit only for small CPU unit cases. The CLI
    always requires unresized 900x1600 images and the fixed 928x1600 model ABI.
    """
    if image.dtype != np.uint8 or image.shape != (*expected_hw, 3):
        raise ValueError(f"Expected uint8 BGR {(*expected_hw, 3)}, got {image.dtype} {image.shape}; no implicit resize")
    if pad_divisor <= 0:
        raise ValueError("pad_divisor must be positive")
    normalized = image.astype(np.float32)
    # MMCV converts its float32 means to float64 for the OpenCV arithmetic.
    cv2.subtract(normalized, MEAN.astype(np.float64).reshape(1, 3), normalized)
    cv2.multiply(normalized, np.ones((1, 3), dtype=np.float64), normalized)
    height, width = expected_hw
    padded = np.zeros(((height + pad_divisor - 1) // pad_divisor * pad_divisor,
                       (width + pad_divisor - 1) // pad_divisor * pad_divisor, 3), dtype=np.float32)
    padded[:height, :width] = normalized
    return np.ascontiguousarray(padded.transpose(2, 0, 1))


def validate_export_report(path):
    if path is None:
        return None
    path = Path(path)
    report = json.loads(path.read_text(encoding="utf-8"))
    normalization = report.get("profile", {}).get("normalization")
    expected = {"mean": [103.53, 116.28, 123.675], "std": [1.0, 1.0, 1.0], "to_rgb": False}
    if report.get("input", {}).get("images") != list(INPUT_SHAPE) or normalization != expected:
        raise ValueError("Export report disagrees with the fixed Base preprocessing contract")
    return {"path": str(path.resolve()), "sha256": file_sha256(path), "config_sha256": report.get("config_sha256")}


def prepare(root, output, *, maximum=32, export_report=None):
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Select a new output directory: {output}")
    contract_evidence = validate_export_report(export_report)
    grouped = read_mini_index(root)
    selected = select_samples(grouped, maximum)
    # Check all selected source paths/metadata before producing calibration files.
    for sample in selected:
        for camera in sample["cameras"]:
            image_path(root, camera["filename"])
            if (camera.get("height"), camera.get("width")) != ORIGINAL_HW:
                raise ValueError(f"Unexpected source image dimensions: {camera['filename']}")
    output.mkdir(parents=True, exist_ok=False)
    samples, files = [], []
    for index, sample in enumerate(selected):
        batch = np.empty(INPUT_SHAPE, dtype=np.float32)
        views = []
        for camera_index, camera in enumerate(sample["cameras"]):
            path = image_path(root, camera["filename"])
            encoded = np.fromfile(path, dtype=np.uint8)
            image = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
            if image is None:
                raise ValueError(f"OpenCV could not decode {path}")
            batch[0, camera_index] = preprocess_bgr(image)
            views.append({"channel": camera["channel"], "sample_data_token": camera["token"],
                          "calibrated_sensor_token": camera["calibrated_sensor_token"],
                          "sensor_token": camera["sensor_token"], "ego_pose_token": camera.get("ego_pose_token"),
                          "timestamp": camera.get("timestamp"), "filename": camera["filename"],
                          "source_sha256": hashlib.sha256(encoded).hexdigest()})
        filename = f"frame_{index:04d}.npy"
        np.save(output / filename, batch, allow_pickle=False)
        files.append(filename)
        samples.append({**{k: sample[k] for k in ("sample_token", "scene_token", "scene_name", "timestamp")},
                        "file": filename, "sha256": file_sha256(output / filename), "cameras": views})
        print(f"[{index + 1}/{len(selected)}] {sample['scene_name']} {sample['sample_token']} -> {filename}", flush=True)
    manifest = {
        "schema_version": 1, "source_kind": "real_preprocessed",
        "dataset": "nuScenes v1.0-mini official mini_train camera key frames",
        "preprocessing": "OpenCV unchanged BGR uint8 -> float32; subtract BGR [103.53,116.28,123.675], std=1; no RGB conversion, resize, flip or augmentation; then bottom zero-pad 900x1600 to 928x1600; N,V,C,H,W",
        "files": files, "input_name": "images", "input_shape": list(INPUT_SHAPE), "input_dtype": "float32",
        "camera_order": list(CAMERA_ORDER),
        "preprocessing_contract": {"mean_bgr": [103.53, 116.28, 123.675], "std": [1.0, 1.0, 1.0],
                                   "to_rgb": False, "original_hw": list(ORIGINAL_HW), "pad_divisor": 32,
                                   "pad_after_normalization": True, "normalized_pad_value": 0.0,
                                   "sources": [CONFIG_SOURCE, CAMERA_SOURCE, PIPELINE_SOURCE],
                                   "export_report": contract_evidence},
        "sampling": {"method": "balanced scene quotas; nearest frames to uniform time-bin centers",
                     "max_samples": maximum, "selected_samples": len(samples), "calibration_split": "mini_train",
                     "calibration_scene_names": list(MINI_TRAIN), "held_out_split": "mini_val",
                     "held_out_scene_names": list(MINI_VAL), "split_source": SPLIT_SOURCE},
        "source_tables": {name: {"file": f"v1.0-mini/{name}.json", "sha256": file_sha256(root / "v1.0-mini" / f"{name}.json")}
                          for name in TABLES},
        "dataset_root": str(root), "opencv_version": cv2.__version__, "samples": samples,
        "limitations": [
            "The ONNX model still uses fixed synthetic all-visible projection and zero CAN-bus; real per-frame geometry is not supplied.",
            "This calibrates real image inputs for that fixed cold graph; head activation statistics may differ from a real-geometry temporal deployment.",
            "Calibration is not nuScenes mAP/NDS evaluation; mini_val is held out and must not be used to fit quantization ranges.",
        ],
    }
    (output / "calibration.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True, help="Directory containing v1.0-mini/ and samples/")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=32)
    parser.add_argument("--export-report", type=Path, help="Check shape/normalization against the actual model export report")
    args = parser.parse_args(argv)
    try:
        manifest = prepare(args.data_root, args.output, maximum=args.max_samples, export_report=args.export_report)
    except Exception as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 1
    print(f"Prepared {len(manifest['files'])} calibration batches: {args.output / 'calibration.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
