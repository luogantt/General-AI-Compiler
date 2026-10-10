from __future__ import annotations

import unittest

import numpy as np

from gac.bevformer_postprocess import (
    NMSFreeDecodeConfig,
    decode_nms_free,
    shift_detections_to_bottom_center,
)
from gac.errors import ValidationError


class NMSFreeDecodeTests(unittest.TestCase):
    def test_topk_label_query_mapping_and_box_denormalization(self):
        config = NMSFreeDecodeConfig(num_classes=2, max_num=3)
        logits = np.asarray([[[0.0, 4.0], [3.0, -1.0], [2.0, 1.0]]], dtype=np.float32)
        boxes = np.zeros((1, 3, 10), dtype=np.float32)
        boxes[0, :, 0] = [1.0, 2.0, 3.0]
        boxes[0, :, 1] = [4.0, 5.0, 6.0]
        boxes[0, :, 4] = [0.5, 1.5, 2.5]
        boxes[0, :, 2] = np.log([2.0, 3.0, 4.0])
        boxes[0, :, 3] = np.log([5.0, 6.0, 7.0])
        boxes[0, :, 5] = np.log([8.0, 9.0, 10.0])
        boxes[0, :, 6] = [0.0, 1.0, 0.0]
        boxes[0, :, 7] = [1.0, 0.0, -1.0]
        boxes[0, :, 8:] = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
        result = decode_nms_free(logits, boxes, config)[0]
        np.testing.assert_array_equal(result.query_indices, [0, 1, 2])
        np.testing.assert_array_equal(result.labels, [1, 0, 0])
        np.testing.assert_allclose(result.boxes[:, :3], [[1, 4, 0.5], [2, 5, 1.5], [3, 6, 2.5]])
        np.testing.assert_allclose(result.boxes[:, 3:6], [[2, 5, 8], [3, 6, 9], [4, 7, 10]], rtol=1e-6)
        np.testing.assert_allclose(result.boxes[:, 7:], [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])

        bottom_center = shift_detections_to_bottom_center(result)
        np.testing.assert_allclose(
            bottom_center.boxes[:, :3],
            [[1, 4, -3.5], [2, 5, -3.0], [3, 6, -2.5]],
            atol=1e-6,
        )
        np.testing.assert_array_equal(bottom_center.boxes[:, 3:], result.boxes[:, 3:])
        np.testing.assert_array_equal(bottom_center.scores, result.scores)
        np.testing.assert_array_equal(bottom_center.labels, result.labels)
        np.testing.assert_array_equal(bottom_center.query_indices, result.query_indices)
        np.testing.assert_allclose(result.boxes[:, 2], [0.5, 1.5, 2.5])

    def test_post_center_range_and_score_threshold_filter(self):
        config = NMSFreeDecodeConfig(
            num_classes=1,
            max_num=3,
            score_threshold=0.8,
            post_center_range=(-1, -1, -1, 1, 1, 1),
        )
        logits = np.asarray([[[4.0], [3.0], [2.0]]], dtype=np.float32)
        boxes = np.zeros((1, 3, 10), dtype=np.float32)
        boxes[0, :, 0] = [0.0, 2.0, 0.5]
        result = decode_nms_free(logits, boxes, config)[0]
        np.testing.assert_array_equal(result.query_indices, [0, 2])

    def test_contract_rejections(self):
        logits = np.zeros((1, 2, 10), dtype=np.float32)
        boxes = np.zeros((1, 2, 10), dtype=np.float32)
        decode_nms_free(logits, boxes, NMSFreeDecodeConfig(max_num=20))
        with self.assertRaisesRegex(ValidationError, "num_classes"):
            decode_nms_free(logits[:, :, :9].copy(), boxes)
        bad = boxes.copy()
        bad[0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValidationError, "finite"):
            decode_nms_free(logits, bad)
        with self.assertRaises(ValidationError):
            NMSFreeDecodeConfig(code_size=7)
        with self.assertRaisesRegex(ValidationError, "10-channel"):
            NMSFreeDecodeConfig(code_size=8)
        with self.assertRaisesRegex(ValidationError, "score_threshold"):
            NMSFreeDecodeConfig(score_threshold=0.0)
        with self.assertRaisesRegex(ValidationError, "max_num"):
            decode_nms_free(
                logits,
                boxes,
                NMSFreeDecodeConfig(num_classes=10, max_num=21),
            )
        with self.assertRaisesRegex(ValidationError, "DecodedDetections"):
            shift_detections_to_bottom_center(object())


if __name__ == "__main__":
    unittest.main()
