"""Tests for all six BEVFormer detection projection layers."""

from __future__ import annotations

import unittest

import numpy as np

from gac.backends.cuda import compile_cuda
from gac.bevformer_detection_head import (
    DetectionHeadConfig,
    build_detection_head_graph,
)
from gac.bevformer_detection_stack import (
    DetectionStackConfig,
    build_detection_stack_graph,
    detection_stack_weight_shapes,
    validate_detection_stack_inputs,
)
from gac.errors import ValidationError
from gac.runtime import reference_run


SMALL_HEAD = DetectionHeadConfig(
    queries=3,
    embed_dim=4,
    classes=2,
    code_size=6,
    point_cloud_range=(-2.0, -3.0, -1.0, 2.0, 5.0, 3.0),
)
SMALL = DetectionStackConfig(head=SMALL_HEAD)


def make_weights(config=SMALL, seed=8610, *, zeros=False):
    rng = np.random.default_rng(seed)
    result = []
    for shapes in detection_stack_weight_shapes(config):
        group = {}
        for name, shape in shapes.items():
            if zeros:
                value = np.zeros(shape, dtype=np.float32)
                if name.endswith("norm_weight"):
                    value.fill(1.0)
            elif name.endswith("norm_weight"):
                value = rng.uniform(0.8, 1.2, shape).astype(np.float32)
            elif name.endswith("norm_bias"):
                value = rng.uniform(-0.1, 0.1, shape).astype(np.float32)
            else:
                value = (rng.standard_normal(shape) * 0.12).astype(np.float32)
            group[name] = np.ascontiguousarray(value)
        result.append(group)
    return tuple(result)


def make_inputs(config=SMALL, seed=3070):
    rng = np.random.default_rng(seed)
    layers = config.num_layers
    batch = config.batch
    queries = config.queries
    return {
        "intermediate_states": np.ascontiguousarray(
            rng.standard_normal(
                (layers, batch, queries, config.embed_dim), dtype=np.float32
            )
        ),
        "initial_reference_points": np.ascontiguousarray(
            rng.uniform(0.1, 0.9, (batch, queries, 3)).astype(np.float32)
        ),
        "intermediate_reference_points": np.ascontiguousarray(
            rng.uniform(0.1, 0.9, (layers, batch, queries, 3)).astype(np.float32)
        ),
        "regressions": np.ascontiguousarray(
            rng.standard_normal(
                (layers, batch, queries, config.code_size), dtype=np.float32
            )
        ),
    }


class DetectionStackTests(unittest.TestCase):
    def test_small_stack_matches_six_independent_heads(self):
        weights = make_weights()
        inputs = make_inputs()
        graph = build_detection_stack_graph(weights, SMALL)
        validate_detection_stack_inputs(graph, inputs)
        actual = reference_run(graph, inputs)

        logits = []
        scores = []
        boxes = []
        for index in range(SMALL.num_layers):
            reference = (
                inputs["initial_reference_points"]
                if index == 0
                else inputs["intermediate_reference_points"][index - 1]
            )
            expected = reference_run(
                build_detection_head_graph(weights[index], SMALL.head),
                {
                    "query": np.ascontiguousarray(
                        inputs["intermediate_states"][index]
                    ),
                    "reference_points": np.ascontiguousarray(reference),
                    "regression": np.ascontiguousarray(
                        inputs["regressions"][index]
                    ),
                },
            )
            logits.append(expected["class_logits"])
            scores.append(expected["class_scores"])
            boxes.append(expected["boxes"])

        expected_logits = np.stack(logits)
        expected_scores = np.stack(scores)
        expected_boxes = np.stack(boxes)
        np.testing.assert_allclose(
            actual["all_class_logits"], expected_logits, rtol=1e-6, atol=1e-6
        )
        np.testing.assert_allclose(
            actual["all_class_scores"], expected_scores, rtol=1e-6, atol=1e-6
        )
        np.testing.assert_allclose(
            actual["all_boxes"], expected_boxes, rtol=1e-6, atol=1e-6
        )
        np.testing.assert_allclose(
            actual["final_class_logits"], expected_logits[-1], rtol=1e-6, atol=1e-6
        )
        np.testing.assert_allclose(
            actual["final_class_scores"], expected_scores[-1], rtol=1e-6, atol=1e-6
        )
        np.testing.assert_allclose(
            actual["final_boxes"], expected_boxes[-1], rtol=1e-6, atol=1e-6
        )

        self.assertTrue(graph.metadata["full_detection_projection_stack"])
        self.assertEqual(graph.metadata["num_layers"], 6)
        self.assertEqual(
            graph.metadata["reference_selection"]["layer0"],
            "initial_reference_points",
        )
        self.assertEqual(
            graph.metadata["reference_selection"]["layers1_to_5"],
            "intermediate_reference_points[layer - 1]",
        )
        self.assertEqual(graph.value_specs["all_class_logits"].shape, (6, 1, 3, 2))
        self.assertEqual(graph.value_specs["all_class_scores"].shape, (6, 1, 3, 2))
        self.assertEqual(graph.value_specs["all_boxes"].shape, (6, 1, 3, 6))

    def test_default_profile_compiles_offline_for_sm86(self):
        config = DetectionStackConfig()
        self.assertEqual(
            (
                config.num_layers,
                config.batch,
                config.queries,
                config.embed_dim,
                config.classes,
                config.code_size,
            ),
            (6, 1, 900, 256, 10, 10),
        )
        graph = build_detection_stack_graph(make_weights(config, zeros=True), config)
        program = compile_cuda(graph)
        self.assertEqual(program.describe()["target"], "cuda-sm86")
        self.assertEqual(
            program.reference_graph.value_specs["all_class_logits"].shape,
            (6, 1, 900, 10),
        )
        self.assertEqual(
            program.reference_graph.value_specs["all_boxes"].shape,
            (6, 1, 900, 10),
        )
        self.assertEqual(graph.metadata["nms_free_topk"], "outside-this-graph")
        self.assertEqual(
            graph.metadata["topk_and_final_nms_free_decode"],
            "outside-this-graph",
        )

    def test_strict_config_weight_and_input_contracts(self):
        weights = make_weights()
        graph = build_detection_stack_graph(weights, SMALL)
        values = make_inputs()
        validate_detection_stack_inputs(graph, values)

        with self.assertRaises(ValidationError):
            DetectionStackConfig(num_layers=5)
        with self.assertRaises(ValidationError):
            DetectionStackConfig(head=object())  # type: ignore[arg-type]
        with self.assertRaises(ValidationError):
            detection_stack_weight_shapes(object())  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValidationError, "exactly 6"):
            build_detection_stack_graph(weights[:-1], SMALL)

        broken = list(weights)
        broken_layer = dict(broken[2])
        broken_layer["cls2_bias"] = np.zeros((3,), dtype=np.float32)
        broken[2] = broken_layer
        with self.assertRaisesRegex(ValidationError, "cls2_bias"):
            build_detection_stack_graph(broken, SMALL)

        out_of_range = dict(values)
        out_of_range["intermediate_reference_points"] = values[
            "intermediate_reference_points"
        ].copy()
        out_of_range["intermediate_reference_points"].flat[0] = 1.1
        with self.assertRaisesRegex(ValidationError, "normalized"):
            validate_detection_stack_inputs(graph, out_of_range)

        nonfinite = dict(values)
        nonfinite["regressions"] = values["regressions"].copy()
        nonfinite["regressions"].flat[0] = np.nan
        with self.assertRaisesRegex(ValidationError, "contiguous and finite"):
            validate_detection_stack_inputs(graph, nonfinite)
        with self.assertRaises(ValidationError):
            validate_detection_stack_inputs(graph, {})


if __name__ == "__main__":
    unittest.main()
