"""Explicit host implementation of BEVFormer's NMS-free coder decode.

The compiler graph intentionally returns fixed-shape query tensors.  This
module performs the variable-length Top-K and post-center-range filtering used
by the official ``NMSFreeCoder``.  Its decoded boxes retain the center-z
convention returned by that coder.  The official ``BEVFormerHead.get_bboxes``
then shifts z down by half the box height before constructing the framework's
bottom-centered box type; :func:`shift_detections_to_bottom_center` exposes
that distinct boundary explicitly.  Keeping these as host stages makes the
CPU boundary visible until TopK/Gather receive CUDA lowering support.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .errors import ValidationError


@dataclass(frozen=True, slots=True)
class NMSFreeDecodeConfig:
    num_classes: int = 10
    max_num: int = 300
    code_size: int = 10
    score_threshold: float | None = None
    post_center_range: tuple[float, float, float, float, float, float] = (
        -61.2,
        -61.2,
        -10.0,
        61.2,
        61.2,
        10.0,
    )

    def __post_init__(self) -> None:
        for name in ("num_classes", "max_num", "code_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValidationError(f"NMS-free decode {name} must be positive")
        if self.code_size != 10:
            raise ValidationError(
                "NMS-free decode requires the frozen 10-channel box code"
            )
        if self.score_threshold is not None:
            value = self.score_threshold
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float, np.integer, np.floating))
                or not math.isfinite(float(value))
                or not 0.0 < float(value) <= 1.0
            ):
                raise ValidationError(
                    "NMS-free decode score_threshold must be finite in (0,1]"
                )
            object.__setattr__(self, "score_threshold", float(value))
        if (
            not isinstance(self.post_center_range, tuple)
            or len(self.post_center_range) != 6
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for value in self.post_center_range
            )
        ):
            raise ValidationError(
                "NMS-free decode post_center_range must contain six finite values"
            )
        low, high = self.post_center_range[:3], self.post_center_range[3:]
        if any(float(upper) <= float(lower) for lower, upper in zip(low, high, strict=True)):
            raise ValidationError(
                "NMS-free decode post_center_range maxima must exceed minima"
            )


@dataclass(frozen=True, slots=True)
class DecodedDetections:
    """One batch item returned by the official coder-level decode.

    ``boxes`` uses ``[x, y, center_z, w, l, h, yaw, vx, vy]``.  Call
    :func:`shift_detections_to_bottom_center` before comparing against the
    tensor passed by official ``BEVFormerHead.get_bboxes`` to ``box_type_3d``.
    """

    boxes: np.ndarray
    scores: np.ndarray
    labels: np.ndarray
    query_indices: np.ndarray

    def __post_init__(self) -> None:
        count = self.scores.shape[0]
        if self.boxes.ndim != 2 or self.boxes.shape != (count, 9):
            raise ValidationError("decoded boxes must have shape [N,9]")
        if self.labels.shape != (count,) or self.query_indices.shape != (count,):
            raise ValidationError("decoded labels and query_indices must have shape [N]")
        if self.boxes.dtype != np.float32 or self.scores.dtype != np.float32:
            raise ValidationError("decoded boxes and scores must be float32")
        if self.labels.dtype != np.int64 or self.query_indices.dtype != np.int64:
            raise ValidationError("decoded labels and query_indices must be int64")


def _sigmoid(value: np.ndarray) -> np.ndarray:
    positive = value >= 0
    result = np.empty_like(value)
    result[positive] = 1.0 / (1.0 + np.exp(-value[positive]))
    exponent = np.exp(value[~positive])
    result[~positive] = exponent / (1.0 + exponent)
    return result


def _denormalize_boxes(boxes: np.ndarray) -> np.ndarray:
    """Match ``projects/.../core/bbox/util.py::denormalize_bbox``."""

    rotation = np.arctan2(boxes[:, 6], boxes[:, 7]).astype(np.float32)
    size = np.exp(boxes[:, [2, 3, 5]]).astype(np.float32)
    parts = [
        boxes[:, 0:2],
        boxes[:, 4:5],
        size,
        rotation[:, None],
    ]
    if boxes.shape[1] > 8:
        parts.append(boxes[:, 8:10])
    decoded = np.concatenate(parts, axis=-1)
    # The frozen profile has velocity and therefore exactly nine decoded dims.
    if decoded.shape[1] != 9:
        raise ValidationError(
            "NMS-free decode requires the frozen 10-channel box code with velocity"
        )
    return np.ascontiguousarray(decoded, dtype=np.float32)


def _topk_indices(scores: np.ndarray, count: int) -> np.ndarray:
    """Return descending Top-K with a documented stable flattened-index tie break."""

    if count == scores.size:
        candidates = np.arange(scores.size, dtype=np.int64)
    else:
        candidates = np.argpartition(-scores, count - 1)[:count].astype(np.int64)
    # lexsort uses the last key as primary: score descending, then index ascending.
    order = np.lexsort((candidates, -scores[candidates]))
    return np.ascontiguousarray(candidates[order], dtype=np.int64)


def decode_nms_free(
    class_logits: np.ndarray,
    box_predictions: np.ndarray,
    config: NMSFreeDecodeConfig | None = None,
) -> tuple[DecodedDetections, ...]:
    """Match ``NMSFreeCoder.decode`` and return center-z variable detections."""

    config = config or NMSFreeDecodeConfig()
    if not isinstance(config, NMSFreeDecodeConfig):
        raise ValidationError("config must be an NMSFreeDecodeConfig")
    for name, value in (
        ("class_logits", class_logits),
        ("box_predictions", box_predictions),
    ):
        if not isinstance(value, np.ndarray):
            raise ValidationError(f"{name} must be a NumPy ndarray")
        if value.dtype != np.float32 or not value.flags.c_contiguous:
            raise ValidationError(f"{name} must be contiguous float32")
        if not np.isfinite(value).all():
            raise ValidationError(f"{name} must contain only finite values")
    if class_logits.ndim != 3 or class_logits.shape[-1] != config.num_classes:
        raise ValidationError(
            "class_logits must have shape [batch,queries,num_classes]"
        )
    if (
        box_predictions.ndim != 3
        or box_predictions.shape[:2] != class_logits.shape[:2]
        or box_predictions.shape[-1] != config.code_size
    ):
        raise ValidationError(
            "box_predictions must have shape [batch,queries,code_size]"
        )
    flattened_score_count = class_logits.shape[1] * config.num_classes
    if config.max_num > flattened_score_count:
        raise ValidationError(
            "NMS-free decode max_num cannot exceed queries * num_classes"
        )

    results: list[DecodedDetections] = []
    lower = np.asarray(config.post_center_range[:3], dtype=np.float32)
    upper = np.asarray(config.post_center_range[3:], dtype=np.float32)
    for batch_index in range(class_logits.shape[0]):
        scores = _sigmoid(class_logits[batch_index]).reshape(-1)
        count = config.max_num
        flat_indices = _topk_indices(scores, count)
        selected_scores = np.ascontiguousarray(scores[flat_indices], dtype=np.float32)
        labels = np.ascontiguousarray(
            flat_indices % config.num_classes, dtype=np.int64
        )
        query_indices = np.ascontiguousarray(
            flat_indices // config.num_classes, dtype=np.int64
        )
        selected = box_predictions[batch_index, query_indices]
        decoded = _denormalize_boxes(selected)

        mask = np.all(decoded[:, :3] >= lower, axis=1) & np.all(
            decoded[:, :3] <= upper, axis=1
        )
        if config.score_threshold is not None:
            threshold = config.score_threshold
            score_mask = selected_scores > np.float32(threshold)
            while not np.any(score_mask):
                threshold *= 0.9
                if threshold < 0.01:
                    score_mask = np.ones_like(mask)
                    break
                score_mask = selected_scores >= np.float32(threshold)
            mask &= score_mask

        results.append(
            DecodedDetections(
                boxes=np.ascontiguousarray(decoded[mask], dtype=np.float32),
                scores=np.ascontiguousarray(selected_scores[mask], dtype=np.float32),
                labels=np.ascontiguousarray(labels[mask], dtype=np.int64),
                query_indices=np.ascontiguousarray(
                    query_indices[mask], dtype=np.int64
                ),
            )
        )
    return tuple(results)


def shift_detections_to_bottom_center(
    detections: DecodedDetections,
) -> DecodedDetections:
    """Match the z-origin adjustment in official ``BEVFormerHead.get_bboxes``.

    The official head applies ``z -= h * 0.5`` after ``NMSFreeCoder.decode``
    and before constructing ``box_type_3d``.  This function returns a copied
    record so the coder-level result and its center-z contract remain intact.
    """

    if not isinstance(detections, DecodedDetections):
        raise ValidationError("detections must be a DecodedDetections record")
    boxes = detections.boxes.copy()
    boxes[:, 2] -= boxes[:, 5] * np.float32(0.5)
    return DecodedDetections(
        boxes=np.ascontiguousarray(boxes, dtype=np.float32),
        scores=detections.scores.copy(),
        labels=detections.labels.copy(),
        query_indices=detections.query_indices.copy(),
    )


__all__ = [
    "DecodedDetections",
    "NMSFreeDecodeConfig",
    "decode_nms_free",
    "shift_detections_to_bottom_center",
]
