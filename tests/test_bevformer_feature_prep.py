from __future__ import annotations

import unittest

import numpy as np

from gac.backends.cuda import compile_cuda
from gac.bevformer_feature_prep import (
    BevFormerFeaturePrepConfig,
    build_bevformer_feature_prep_graph,
    feature_prep_weight_shapes,
    validate_feature_prep_inputs,
)
from gac.errors import ValidationError
from gac.runtime import reference_run


def make_weights(config: BevFormerFeaturePrepConfig):
    result = {}
    cursor = 1
    for name, shape in feature_prep_weight_shapes(config).items():
        size = int(np.prod(shape))
        result[name] = np.ascontiguousarray(
            np.arange(cursor, cursor + size, dtype=np.float32).reshape(shape) / 100.0
        )
        cursor += size
    return result


class BevFormerFeaturePrepTests(unittest.TestCase):
    def setUp(self):
        self.config = BevFormerFeaturePrepConfig(
            cameras=2, embed_dim=4, feature_h=2, feature_w=3
        )
        self.weights = make_weights(self.config)

    def test_layout_embeddings_and_reference_semantics(self):
        graph = build_bevformer_feature_prep_graph(self.weights, self.config)
        feature = np.arange(
            np.prod(graph.inputs["fpn_feature"].shape), dtype=np.float32
        ).reshape(graph.inputs["fpn_feature"].shape)
        actual = reference_run(graph, {"fpn_feature": feature})["image_value"]
        expected = feature.reshape(1, 2, 4, 2, 3)
        expected = expected.transpose(0, 1, 3, 4, 2).reshape(1, 2, 6, 4)
        expected = (
            expected
            + self.weights["camera_embeddings"].reshape(1, 2, 1, 4)
            + self.weights["level_embedding"].reshape(1, 1, 1, 4)
        )
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(graph.value_specs["image_value"].shape, (1, 2, 6, 4))
        self.assertEqual(graph.metadata["feature_levels"], 1)

    def test_default_profile_compiles_for_sm86(self):
        config = BevFormerFeaturePrepConfig()
        weights = {
            name: np.zeros(shape, dtype=np.float32)
            for name, shape in feature_prep_weight_shapes(config).items()
        }
        program = compile_cuda(build_bevformer_feature_prep_graph(weights, config))
        self.assertEqual(len(program.launches), 4)
        self.assertEqual(len(program.describe()["zero_copy_views"]), 1)
        self.assertIn("__global__", program.source)

    def test_strict_weights_and_inputs(self):
        graph = build_bevformer_feature_prep_graph(self.weights, self.config)
        value = np.zeros(graph.inputs["fpn_feature"].shape, dtype=np.float32)
        validate_feature_prep_inputs(graph, {"fpn_feature": value})
        with self.assertRaisesRegex(ValidationError, "weight names"):
            build_bevformer_feature_prep_graph(
                {"camera_embeddings": self.weights["camera_embeddings"]},
                self.config,
            )
        with self.assertRaisesRegex(ValidationError, "requires"):
            validate_feature_prep_inputs(
                graph, {"fpn_feature": value.astype(np.float16)}
            )


if __name__ == "__main__":
    unittest.main()
