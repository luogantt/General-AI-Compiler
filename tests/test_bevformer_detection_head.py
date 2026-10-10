from __future__ import annotations

import unittest

import numpy as np

from gac.backends.cuda import compile_cuda
from gac.bevformer_detection_head import (
    DetectionHeadConfig,
    build_detection_head_graph,
    detection_head_weight_shapes,
    validate_detection_head_inputs,
)
from gac.errors import ValidationError
from gac.runtime import reference_run


SMALL = DetectionHeadConfig(
    queries=3,
    embed_dim=4,
    classes=2,
    code_size=6,
    point_cloud_range=(-2.0, -3.0, -1.0, 2.0, 5.0, 3.0),
)


def layer_norm(x, weight, bias, eps):
    mean = x.mean(axis=-1, keepdims=True)
    variance = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(variance + eps) * weight + bias


class DetectionHeadTests(unittest.TestCase):
    def test_small_graph_matches_independent_oracle(self):
        rng = np.random.default_rng(3070)
        weights = {
            name: rng.normal(0, 0.1, shape).astype(np.float32)
            for name, shape in detection_head_weight_shapes(SMALL).items()
        }
        inputs = {
            "query": rng.normal(size=(1, 3, 4)).astype(np.float32),
            "reference_points": rng.uniform(0.1, 0.9, (1, 3, 3)).astype(np.float32),
            "regression": rng.normal(size=(1, 3, 6)).astype(np.float32),
        }
        graph = build_detection_head_graph(weights, SMALL)
        validate_detection_head_inputs(graph, inputs)
        actual = reference_run(compile_cuda(graph).reference_graph, inputs)

        hidden0 = inputs["query"] @ weights["cls0_weight"] + weights["cls0_bias"]
        hidden0 = np.maximum(layer_norm(hidden0, weights["cls0_norm_weight"], weights["cls0_norm_bias"], SMALL.norm_epsilon), 0)
        hidden1 = hidden0 @ weights["cls1_weight"] + weights["cls1_bias"]
        hidden1 = np.maximum(layer_norm(hidden1, weights["cls1_norm_weight"], weights["cls1_norm_bias"], SMALL.norm_epsilon), 0)
        logits = hidden1 @ weights["cls2_weight"] + weights["cls2_bias"]
        reference_logits = np.log(inputs["reference_points"] / (1 - inputs["reference_points"]))
        xy = 1 / (1 + np.exp(-(inputs["regression"][..., :2] + reference_logits[..., :2])))
        xy = xy * np.asarray([4.0, 8.0], np.float32) + np.asarray([-2.0, -3.0], np.float32)
        z = 1 / (1 + np.exp(-(inputs["regression"][..., 4:5] + reference_logits[..., 2:3])))
        z = z * np.float32(4.0) - np.float32(1.0)
        boxes = np.concatenate((xy, inputs["regression"][..., 2:4], z, inputs["regression"][..., 5:]), axis=-1)
        np.testing.assert_allclose(actual["class_logits"], logits, rtol=2e-5, atol=2e-5)
        np.testing.assert_allclose(actual["class_scores"], 1 / (1 + np.exp(-logits)), rtol=2e-5, atol=2e-5)
        np.testing.assert_allclose(actual["boxes"], boxes, rtol=2e-5, atol=2e-5)

    def test_default_contract_compiles_and_rejects_bad_reference(self):
        config = DetectionHeadConfig()
        weights = {
            name: np.zeros(shape, np.float32)
            for name, shape in detection_head_weight_shapes(config).items()
        }
        graph = build_detection_head_graph(weights, config)
        program = compile_cuda(graph)
        self.assertEqual(program.graph.value_specs["class_scores"].shape, (1, 900, 10))
        self.assertEqual(program.graph.value_specs["boxes"].shape, (1, 900, 10))
        self.assertEqual(graph.metadata["nms_free_topk"], "outside-this-graph")
        values = {
            "query": np.zeros((1, 900, 256), np.float32),
            "reference_points": np.zeros((1, 900, 3), np.float32),
            "regression": np.zeros((1, 900, 10), np.float32),
        }
        validate_detection_head_inputs(graph, values)
        values["reference_points"][0, 0, 0] = 2
        with self.assertRaisesRegex(ValidationError, "normalized"):
            validate_detection_head_inputs(graph, values)


if __name__ == "__main__":
    unittest.main()
