"""Tests for the framework-independent BEVFormer ResNet/FPN importer."""

from __future__ import annotations

import unittest

import numpy as np

from gac.frontends.bevformer_backbone import (
    BEVFORMER_R50_FPN_PROFILE,
    MMDET_CHECKPOINT_KEYS,
    BevFormerBackboneError,
    ResNetFpnProfile,
    build_bevformer_backbone_graph,
    required_checkpoint_specs,
    validate_checkpoint_weights,
)
from gac.passes import ShapeInferencePass
from gac.runtime import reference_run


SMALL_PROFILE = ResNetFpnProfile(
    name="unit-test-bottleneck-fpn",
    batch=1,
    input_channels=3,
    input_hw=(16, 20),
    stem_channels=2,
    stage_planes=(2, 3, 4, 5),
    blocks=(1, 1, 1, 1),
    expansion=2,
    fpn_channels=3,
)


def small_weights(*, final_bias: float = 0.0) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for spec in required_checkpoint_specs(SMALL_PROFILE):
        if spec.role == "bn-scale" or spec.role == "bn-running-variance":
            value = np.ones(spec.shape, dtype=spec.dtype)
        elif spec.key == MMDET_CHECKPOINT_KEYS.bias(
            MMDET_CHECKPOINT_KEYS.fpn_conv()
        ):
            value = np.full(spec.shape, final_bias, dtype=spec.dtype)
        else:
            value = np.zeros(spec.shape, dtype=spec.dtype)
        result[spec.key] = np.ascontiguousarray(value)
    return result


class ProfileAndKeyTests(unittest.TestCase):
    def test_production_profile_is_frozen_for_bevformer_tiny(self):
        profile = BEVFORMER_R50_FPN_PROFILE
        self.assertEqual(profile.batch, 6)
        self.assertEqual(profile.input_shape, (6, 3, 480, 800))
        self.assertEqual(profile.blocks, (3, 4, 6, 3))
        self.assertEqual(profile.stage_planes, (64, 128, 256, 512))
        self.assertEqual(profile.c5_channels, 2048)
        self.assertEqual(profile.output_shape, (6, 256, 15, 25))
        self.assertEqual(profile.input_dtype, "float16")
        self.assertEqual(profile.parameter_dtype, "float32")

        specs = required_checkpoint_specs(profile)
        self.assertEqual(len(specs), 269)
        self.assertEqual(specs[0].key, "img_backbone.conv1.weight")
        self.assertEqual(specs[0].shape, (64, 3, 7, 7))
        by_key = {spec.key: spec for spec in specs}
        self.assertEqual(
            by_key["img_backbone.layer4.2.conv3.weight"].shape,
            (2048, 512, 1, 1),
        )
        self.assertEqual(
            by_key["img_neck.lateral_convs.0.conv.weight"].shape,
            (256, 2048, 1, 1),
        )
        self.assertEqual(
            by_key["img_neck.fpn_convs.0.conv.weight"].shape,
            (256, 256, 3, 3),
        )

    def test_checkpoint_key_map_matches_mmdet_convention(self):
        keys = MMDET_CHECKPOINT_KEYS
        self.assertEqual(keys.stem_bn(), "img_backbone.bn1")
        self.assertEqual(
            keys.block_conv(3, 5, 2), "img_backbone.layer3.5.conv2"
        )
        self.assertEqual(
            keys.downsample_bn(2, 0), "img_backbone.layer2.0.downsample.1"
        )
        self.assertEqual(
            keys.lateral_conv(), "img_neck.lateral_convs.0.conv"
        )


class SmallGraphTests(unittest.TestCase):
    def test_full_small_graph_shape_residual_downsample_and_provenance(self):
        weights = small_weights(final_bias=2.25)
        weights["pts_bbox_head.unrelated.weight"] = np.asarray(
            [np.nan], dtype=np.float32
        )
        weights["img_backbone.bn1.num_batches_tracked"] = np.asarray(
            100, dtype=np.int64
        )
        graph = build_bevformer_backbone_graph(weights, SMALL_PROFILE)

        self.assertEqual(graph.outputs, ("fpn_c5",))
        self.assertEqual(graph.value_specs["fpn_c5"].shape, (1, 3, 1, 1))
        self.assertEqual(graph.value_specs["fpn_c5"].dtype, "float32")
        self.assertEqual(graph.inputs["images"].dtype, "float16")
        self.assertEqual(
            graph.metadata["ignored_checkpoint_keys"],
            [
                "img_backbone.bn1.num_batches_tracked",
                "pts_bbox_head.unrelated.weight",
            ],
        )
        self.assertNotIn("pts_bbox_head.unrelated.weight", graph.constants)

        allowed = {
            "Conv2D",
            "BatchNorm",
            "ReLU",
            "Add",
            "MaxPool2D",
            "Cast",
        }
        self.assertLessEqual({node.op for node in graph.nodes}, allowed)
        self.assertEqual(graph.nodes[0].op, "Cast")
        self.assertEqual(graph.nodes[0].source_nodes, ("bevformer_export.images.cast",))

        additions = [node for node in graph.nodes if node.op == "Add"]
        self.assertEqual(len(additions), sum(SMALL_PROFILE.blocks))
        self.assertEqual(
            additions[0].source_nodes, ("img_backbone.layer1.0",)
        )
        for addition in additions:
            left, right = addition.inputs
            self.assertEqual(
                graph.value_specs[left].shape, graph.value_specs[right].shape
            )

        downsample_convs = [
            node
            for node in graph.nodes
            if node.op == "Conv2D" and ".downsample.0" in node.source_nodes[0]
        ]
        self.assertEqual(len(downsample_convs), 4)
        self.assertEqual(downsample_convs[0].attrs["strides"], [1, 1])
        self.assertEqual(downsample_convs[1].attrs["strides"], [2, 2])
        self.assertEqual(
            graph.nodes[-1].source_nodes,
            ("img_neck.fpn_convs.0.conv",),
        )

        report = ShapeInferencePass().run(graph)
        self.assertFalse(report.changed)
        graph.validate()

    def test_small_graph_executes_end_to_end_with_numpy(self):
        graph = build_bevformer_backbone_graph(
            small_weights(final_bias=1.75), SMALL_PROFILE
        )
        images = np.linspace(
            -1,
            1,
            np.prod(SMALL_PROFILE.input_shape),
            dtype=np.float16,
        ).reshape(SMALL_PROFILE.input_shape)
        result = reference_run(graph, {"images": images})["fpn_c5"]
        self.assertEqual(result.shape, SMALL_PROFILE.output_shape)
        self.assertEqual(result.dtype, np.float32)
        np.testing.assert_array_equal(
            result, np.full(SMALL_PROFILE.output_shape, 1.75, dtype=np.float32)
        )

    def test_weights_are_reused_without_coercion_or_copy(self):
        weights = small_weights()
        selected = validate_checkpoint_weights(weights, SMALL_PROFILE)
        first_key = required_checkpoint_specs(SMALL_PROFILE)[0].key
        self.assertIs(selected[first_key], weights[first_key])
        graph = build_bevformer_backbone_graph(weights, SMALL_PROFILE)
        self.assertIs(graph.constants[first_key], weights[first_key])


class CheckpointValidationTests(unittest.TestCase):
    def test_missing_key_has_exact_diagnostic(self):
        weights = small_weights()
        missing = "img_backbone.layer2.0.conv2.weight"
        del weights[missing]
        with self.assertRaisesRegex(
            BevFormerBackboneError, r"missing required tensors.*layer2\.0\.conv2\.weight"
        ):
            build_bevformer_backbone_graph(weights, SMALL_PROFILE)

    def test_module_prefix_has_actionable_diagnostic(self):
        weights = {
            f"module.{key}": value for key, value in small_weights().items()
        }
        with self.assertRaisesRegex(BevFormerBackboneError, "strip that prefix"):
            validate_checkpoint_weights(weights, SMALL_PROFILE)

    def test_rejects_wrong_shape_dtype_nonfinite_noncontiguous_and_object(self):
        specs = required_checkpoint_specs(SMALL_PROFILE)
        first = specs[0]
        cases: list[tuple[str, object, str]] = []
        cases.append(
            (
                "shape",
                np.zeros((first.shape[0] + 1, *first.shape[1:]), dtype=np.float32),
                "has shape",
            )
        )
        cases.append(("dtype", np.zeros(first.shape, dtype=np.float16), "has dtype"))
        not_finite = np.zeros(first.shape, dtype=np.float32)
        not_finite.flat[0] = np.inf
        cases.append(("finite", not_finite, "NaN or infinity"))
        backing = np.zeros((*first.shape[:-1], first.shape[-1] * 2), dtype=np.float32)
        not_contiguous = backing[..., ::2]
        self.assertEqual(not_contiguous.shape, first.shape)
        self.assertFalse(not_contiguous.flags.c_contiguous)
        cases.append(("contiguous", not_contiguous, "C-contiguous"))
        cases.append(("object", object(), "must be a NumPy array"))

        for name, replacement, message in cases:
            with self.subTest(name=name):
                weights = small_weights()
                weights[first.key] = replacement  # type: ignore[assignment]
                with self.assertRaisesRegex(BevFormerBackboneError, message):
                    validate_checkpoint_weights(weights, SMALL_PROFILE)

    def test_non_string_checkpoint_key_is_rejected(self):
        weights: dict[object, np.ndarray] = dict(small_weights())
        weights[7] = np.zeros((1,), dtype=np.float32)
        with self.assertRaisesRegex(BevFormerBackboneError, "keys must all be strings"):
            validate_checkpoint_weights(weights, SMALL_PROFILE)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
