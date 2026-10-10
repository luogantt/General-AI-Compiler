from __future__ import annotations

from dataclasses import replace
import unittest

import numpy as np

from gac.backends.cuda import compile_cuda
from gac.bevformer_decoder_pipeline import build_bevformer_decoder_pipeline_graph
from gac.bevformer_detection_stack import DetectionStackConfig
from gac.bevformer_encoder_inputs import build_bevformer_encoder_input_graph
from gac.bevformer_encoder_stack import build_bevformer_encoder_stack_graph
from gac.bevformer_feature_prep import build_bevformer_feature_prep_graph
from gac.bevformer_motion import BevFormerMotionConfig, build_bevformer_motion_graph
from gac.bevformer_step_pipeline import (
    BevFormerStepPipelineConfig,
    build_bevformer_step_pipeline_graph,
    step_pipeline_weight_shapes,
    validate_step_pipeline_inputs,
)
from gac.errors import ValidationError
from gac.frontends.bevformer_backbone import build_bevformer_backbone_graph
from gac.bevformer_runtime import BevFormerRuntime, BevFormerStepContract
from gac.runtime import reference_run
from tests.test_bevformer_decoder_pipeline import (
    make_weights as make_decoder_weights,
    small_config as small_decoder_config,
)
from tests.test_bevformer_history_pipeline import (
    random_weights as make_encoder_weight_tuple,
    small_config as small_history_config,
)


def small_config() -> BevFormerStepPipelineConfig:
    history = small_history_config()
    history = replace(
        history,
        backbone=replace(history.backbone, input_dtype="float16"),
        encoder_inputs=replace(history.encoder_inputs, can_bus_dim=18),
    )

    decoder = small_decoder_config()
    cross = replace(
        decoder.decoder.layer.cross_attention, bev_h=1, bev_w=2
    )
    decoder_stack = replace(
        decoder.decoder,
        layer=replace(decoder.decoder.layer, cross_attention=cross),
        refinement=replace(decoder.decoder.refinement, code_size=10),
    )
    detection = DetectionStackConfig(
        head=replace(
            decoder.detection.head,
            classes=10,
            code_size=10,
            point_cloud_range=history.encoder_stack.layer.projection.point_cloud_range,
        )
    )
    decoder = replace(
        decoder, decoder=decoder_stack, detection=detection
    )
    projection = history.encoder_stack.layer.projection
    pc = projection.point_cloud_range
    return BevFormerStepPipelineConfig(
        encoder=history,
        motion=BevFormerMotionConfig(
            bev_h=history.encoder_inputs.bev_h,
            bev_w=history.encoder_inputs.bev_w,
            embed_dim=history.encoder_inputs.embed_dim,
            grid_length_y=(pc[4] - pc[1]) / history.encoder_inputs.bev_h,
            grid_length_x=(pc[3] - pc[0]) / history.encoder_inputs.bev_w,
            rotate_center=(1.0, 0.5),
        ),
        decoder=decoder,
    )


def make_weights(config: BevFormerStepPipelineConfig):
    encoder_tuple = make_encoder_weight_tuple(config.encoder)
    encoder = dict(
        zip(
            (
                "backbone",
                "feature_prep",
                "encoder_inputs",
                "temporal",
                "spatial",
                "layer",
            ),
            encoder_tuple,
            strict=True,
        )
    )
    return encoder, make_decoder_weights(config.decoder)


def runtime_inputs(config: BevFormerStepPipelineConfig, *, warm: bool):
    rng = np.random.default_rng(9107 if warm else 9103)
    images_shape = (
        config.encoder.feature_prep.batch,
        config.encoder.feature_prep.cameras,
        config.encoder.backbone.input_channels,
        *config.encoder.backbone.input_hw,
    )
    can_bus = np.zeros((1, 18), dtype=np.float32)
    # Cold frames retain orientation/absolute-heading fields for the query MLP.
    can_bus[0, 3:7] = np.asarray([0.1, -0.2, 0.3, 0.9], np.float32)
    can_bus[0, -2] = np.float32(0.31)
    if warm:
        can_bus[0, 0:2] = np.asarray([0.25, -0.125], np.float32)
        can_bus[0, -1] = np.float32(7.0)
    return {
        "images": np.ascontiguousarray(
            rng.normal(0.0, 0.2, images_shape).astype(np.float16)
        ),
        "lidar2img": np.ascontiguousarray(
            np.broadcast_to(np.eye(4, dtype=np.float32), (1, 2, 4, 4)).copy()
        ),
        "can_bus_delta": can_bus,
        "prev_bev": np.ascontiguousarray(
            rng.normal(
                0.0,
                0.1,
                (
                    1,
                    config.encoder.encoder_stack.queries,
                    config.encoder.encoder_stack.embed_dim,
                ),
            ).astype(np.float16)
        ),
        "has_prev_bev": np.asarray([1 if warm else 0], dtype=np.int32),
    }


def staged_reference(config, encoder_weights, decoder_weights, values):
    execution_profile = replace(config.encoder.backbone, input_dtype="float32")
    backbone = build_bevformer_backbone_graph(
        encoder_weights["backbone"], execution_profile
    )
    fpn = reference_run(
        backbone,
        {
            "images": values["images"].astype(np.float32).reshape(
                execution_profile.input_shape
            )
        },
    )["fpn_c5"]
    image_value = reference_run(
        build_bevformer_feature_prep_graph(
            encoder_weights["feature_prep"], config.encoder.feature_prep
        ),
        {"fpn_feature": fpn},
    )["image_value"]
    initialized = reference_run(
        build_bevformer_encoder_input_graph(
            encoder_weights["encoder_inputs"], config.encoder.encoder_inputs
        ),
        {"can_bus_delta": values["can_bus_delta"]},
    )
    moved = reference_run(
        build_bevformer_motion_graph(config.motion),
        {
            "prev_bev": values["prev_bev"].astype(np.float32),
            "can_bus_delta": values["can_bus_delta"],
        },
    )
    if values["has_prev_bev"][0]:
        selected_prev = moved["rotated_prev_bev"]
        selected_reference = moved["shifted_reference_points"]
    else:
        selected_prev = initialized["query"]
        selected_reference = initialized["reference_points"]
    warm = bool(values["has_prev_bev"][0])
    stack_config = (
        config.encoder.encoder_stack
        if warm
        else replace(config.encoder.encoder_stack, history_mode=False)
    )
    stack_inputs = {
        "query": initialized["query"],
        "query_pos": initialized["query_pos"],
        "spatial_query_pos": initialized["spatial_query_pos"],
        "reference_points": initialized["reference_points"],
        "image_value": image_value,
        "lidar2img": values["lidar2img"],
    }
    if warm:
        stack_inputs["temporal_value"] = np.ascontiguousarray(
            np.concatenate((selected_prev, initialized["query"]), axis=0)
        )
        stack_inputs["prev_reference_points"] = selected_reference
    encoded = reference_run(
        build_bevformer_encoder_stack_graph(
            encoder_weights["temporal"],
            encoder_weights["spatial"],
            encoder_weights["layer"],
            stack_config,
        ),
        stack_inputs,
    )["output"]
    decoded = reference_run(
        build_bevformer_decoder_pipeline_graph(
            decoder_weights["query_init"],
            decoder_weights["self_attention"],
            decoder_weights["cross_attention"],
            decoder_weights["layer"],
            decoder_weights["refinement"],
            decoder_weights["detection"],
            config.decoder,
        ),
        {"value": encoded},
    )
    return {
        "cls_logits": decoded["final_class_logits"],
        "bbox_regression": decoded["final_boxes"],
        "next_bev": encoded.astype(np.float16),
    }


class BevFormerStepPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = small_config()
        cls.encoder_weights, cls.decoder_weights = make_weights(cls.config)
        cls.graph = build_bevformer_step_pipeline_graph(
            cls.encoder_weights, cls.decoder_weights, cls.config
        )

    def test_cold_and_warm_match_independent_staged_reference(self):
        for warm in (False, True):
            with self.subTest(warm=warm):
                values = runtime_inputs(self.config, warm=warm)
                validate_step_pipeline_inputs(self.graph, values)
                expected = staged_reference(
                    self.config,
                    self.encoder_weights,
                    self.decoder_weights,
                    values,
                )
                actual = reference_run(self.graph, values)
                np.testing.assert_allclose(
                    actual["cls_logits"], expected["cls_logits"],
                    rtol=4e-5, atol=4e-5,
                )
                np.testing.assert_allclose(
                    actual["bbox_regression"], expected["bbox_regression"],
                    rtol=4e-5, atol=4e-5,
                )
                np.testing.assert_array_equal(
                    actual["next_bev"], expected["next_bev"]
                )

    def test_public_abi_and_cold_or_warm_temporal_queues(self):
        self.assertEqual(
            tuple(self.graph.inputs),
            (
                "images",
                "lidar2img",
                "can_bus_delta",
                "prev_bev",
                "has_prev_bev",
            ),
        )
        self.assertEqual(
            tuple(self.graph.outputs),
            ("cls_logits", "bbox_regression", "next_bev"),
        )
        self.assertEqual(self.graph.inputs["images"].dtype, "float16")
        self.assertEqual(self.graph.inputs["prev_bev"].dtype, "float16")
        self.assertEqual(self.graph.inputs["has_prev_bev"].dtype, "int32")
        self.assertEqual(self.graph.value_specs["cls_logits"].shape, (1, 3, 10))
        self.assertEqual(
            self.graph.value_specs["bbox_regression"].shape, (1, 3, 10)
        )
        self.assertEqual(self.graph.value_specs["next_bev"].dtype, "float16")
        nodes = {node.id: node for node in self.graph.nodes}
        self.assertEqual(
            nodes["select_previous_bev"].inputs[0], "has_prev_bev"
        )
        self.assertEqual(
            nodes["select_previous_reference_points"].inputs[0],
            "has_prev_bev",
        )
        queues = [
            node for node in self.graph.nodes
            if node.id == "build_shared_temporal_queue"
        ]
        self.assertEqual(len(queues), 1)
        for index in range(3):
            cold_queue = nodes[f"encoder.layer{index}.cold_temporal_value_concat"]
            expected_query = (
                "initial_bev_query" if index == 0 else f"encoder.layer{index - 1}.output"
            )
            self.assertEqual(cold_queue.inputs, (expected_query, expected_query))
            selector = nodes[f"encoder.layer{index}.select_temporal_value"]
            self.assertEqual(selector.inputs[0], "has_prev_bev")
            self.assertEqual(selector.inputs[1], "shared_temporal_value")
            self.assertEqual(
                selector.inputs[2], f"encoder.layer{index}.cold_temporal_value"
            )
            self.assertEqual(
                nodes[
                    f"encoder.layer{index}.temporal.value_projection"
                ].inputs[0],
                f"encoder.layer{index}.selected_temporal_value",
            )

    def test_default_profile_has_exact_deployment_shapes(self):
        config = BevFormerStepPipelineConfig()
        self.assertEqual(config.encoder.backbone.input_dtype, "float16")
        self.assertEqual(config.motion.can_bus_dim, 18)
        self.assertEqual(
            (
                config.decoder.queries,
                config.decoder.classes,
                config.decoder.code_size,
            ),
            (900, 10, 10),
        )
        self.assertEqual(
            (
                config.encoder.encoder_stack.batch,
                config.encoder.encoder_stack.queries,
                config.encoder.encoder_stack.embed_dim,
            ),
            (1, 2500, 256),
        )

    def test_small_graph_compiles_offline_for_sm86(self):
        program = compile_cuda(self.graph)
        self.assertEqual(program.describe()["target"], "cuda-sm86")
        self.assertGreater(len(program.launches), 200)
        self.assertEqual(
            set(program.reference_graph.outputs),
            {"cls_logits", "bbox_regression", "next_bev"},
        )
        self.assertIn("int* p0", program.source)
        optimized_ids = {node.id for node in program.graph.nodes}
        # Only the final raw head is an ABI output.  Earlier classification
        # heads are dead, while all six refinement branches stay live because
        # each one supplies the next decoder layer's reference points.
        for index in range(5):
            self.assertNotIn(
                f"decoder.detection.layer{index}.head.cls0_matmul",
                optimized_ids,
            )
        self.assertIn(
            "decoder.detection.layer5.head.cls0_matmul", optimized_ids
        )
        for index in range(6):
            self.assertIn(
                f"decoder.decoder.layer{index}.refinement.fc0_matmul",
                optimized_ids,
            )

    def test_stateful_runtime_drives_cold_warm_and_scene_reset_steps(self):
        graph = self.graph

        class ReferenceStepSession:
            def __init__(self):
                self.calls = []

            def run(self, inputs):
                validate_step_pipeline_inputs(graph, inputs)
                self.calls.append(
                    {name: value.copy() for name, value in inputs.items()}
                )
                return reference_run(graph, inputs)

        config = self.config
        contract = BevFormerStepContract(
            cameras=config.encoder.feature_prep.cameras,
            channels=config.encoder.backbone.input_channels,
            image_h=config.encoder.backbone.input_hw[0],
            image_w=config.encoder.backbone.input_hw[1],
            bev_queries=config.encoder.encoder_stack.queries,
            embed_dim=config.encoder.encoder_stack.embed_dim,
            object_queries=config.decoder.queries,
            classes=config.decoder.classes,
            bbox_dims=config.decoder.code_size,
            can_bus_dims=config.motion.can_bus_dim,
        )
        session = ReferenceStepSession()
        runtime = BevFormerRuntime(session, contract=contract)
        sample = runtime_inputs(config, warm=False)

        def absolute_can_bus(position, heading, angle):
            value = np.zeros(contract.can_bus_shape, dtype=np.float32)
            value[0, :3] = np.asarray(position, dtype=np.float32)
            value[0, 3:7] = np.asarray([0.1, -0.2, 0.3, 0.9], np.float32)
            value[0, -2] = np.float32(heading)
            value[0, -1] = np.float32(angle)
            return value

        first = runtime.step(
            sample["images"],
            sample["lidar2img"],
            absolute_can_bus((10.0, 20.0, 2.0), 0.31, 30.0),
            "scene-a",
        )
        self.assertTrue(first.cold_start)
        self.assertEqual(first.frame_index, 1)
        np.testing.assert_array_equal(session.calls[0]["has_prev_bev"], [0])
        np.testing.assert_array_equal(
            session.calls[0]["can_bus_delta"][0, :3], 0
        )
        self.assertEqual(session.calls[0]["can_bus_delta"][0, -1], 0)
        self.assertAlmostEqual(
            float(session.calls[0]["can_bus_delta"][0, -2]),
            0.31,
            places=6,
        )

        second = runtime.step(
            sample["images"],
            sample["lidar2img"],
            absolute_can_bus((10.25, 19.5, 2.0), 0.33, 37.0),
            "scene-a",
        )
        self.assertFalse(second.cold_start)
        self.assertEqual(second.frame_index, 2)
        np.testing.assert_array_equal(session.calls[1]["has_prev_bev"], [1])
        np.testing.assert_allclose(
            session.calls[1]["can_bus_delta"][0, :3],
            [0.25, -0.5, 0.0],
        )
        self.assertEqual(session.calls[1]["can_bus_delta"][0, -1], 7.0)
        np.testing.assert_array_equal(session.calls[1]["prev_bev"], first.next_bev)

        reset = runtime.step(
            sample["images"],
            sample["lidar2img"],
            absolute_can_bus((99.0, 1.0, 1.0), 1.2, 180.0),
            "scene-b",
        )
        self.assertTrue(reset.cold_start)
        self.assertEqual(reset.frame_index, 1)
        np.testing.assert_array_equal(session.calls[2]["prev_bev"], 0)
        self.assertEqual(reset.cls_logits.shape, contract.cls_shape)
        self.assertEqual(reset.bbox_regression.shape, contract.bbox_shape)
        self.assertEqual(reset.next_bev.shape, contract.prev_bev_shape)

    def test_inventory_and_component_profile_contracts(self):
        inventory = step_pipeline_weight_shapes(self.config)
        self.assertEqual(set(inventory), {"encoder", "decoder"})
        self.assertEqual(set(inventory["encoder"]), {
            "backbone", "feature_prep", "encoder_inputs",
            "temporal", "spatial", "layer",
        })
        self.assertEqual(len(inventory["encoder"]["temporal"]), 3)
        self.assertEqual(set(inventory["decoder"]), {
            "query_init", "self_attention", "cross_attention", "layer",
            "refinement", "detection",
        })
        self.assertEqual(len(inventory["decoder"]["self_attention"]), 6)
        self.assertEqual(len(inventory["decoder"]["detection"]), 6)

        bad_encoder = dict(self.encoder_weights)
        bad_encoder.pop("backbone")
        with self.assertRaisesRegex(ValidationError, "weight groups"):
            build_bevformer_step_pipeline_graph(
                bad_encoder, self.decoder_weights, self.config
            )
        with self.assertRaisesRegex(ValidationError, "grid_length_x"):
            replace(
                self.config,
                motion=replace(self.config.motion, grid_length_x=1.0),
            )
        with self.assertRaisesRegex(ValidationError, "image input dtype"):
            replace(
                self.config,
                encoder=replace(
                    self.config.encoder,
                    backbone=replace(
                        self.config.encoder.backbone, input_dtype="float32"
                    ),
                ),
            )
        with self.assertRaisesRegex(ValidationError, "10-channel box code"):
            replace(
                self.config,
                decoder=replace(
                    self.config.decoder,
                    decoder=replace(
                        self.config.decoder.decoder,
                        refinement=replace(
                            self.config.decoder.decoder.refinement,
                            code_size=8,
                        ),
                    ),
                    detection=replace(
                        self.config.decoder.detection,
                        head=replace(
                            self.config.decoder.detection.head,
                            code_size=8,
                        ),
                    ),
                ),
            )
        with self.assertRaises(ValidationError):
            step_pipeline_weight_shapes(object())  # type: ignore[arg-type]

    def test_runtime_abi_is_strict_and_cold_motion_fields_are_zero(self):
        values = runtime_inputs(self.config, warm=False)
        validate_step_pipeline_inputs(self.graph, values)
        self.assertNotEqual(values["can_bus_delta"][0, -2], 0.0)

        cases = []
        missing = dict(values)
        missing.pop("prev_bev")
        cases.append((missing, "input names"))
        wrong_dtype = dict(values)
        wrong_dtype["prev_bev"] = values["prev_bev"].astype(np.float32)
        cases.append((wrong_dtype, "float16"))
        wrong_shape = dict(values)
        wrong_shape["has_prev_bev"] = np.zeros((1, 1), dtype=np.int32)
        cases.append((wrong_shape, "requires"))
        noncontiguous = dict(values)
        backing = np.zeros((1, 2, 8), dtype=np.float16)
        noncontiguous["prev_bev"] = backing[..., ::2]
        cases.append((noncontiguous, "C-contiguous"))
        nonfinite = dict(values)
        nonfinite["lidar2img"] = values["lidar2img"].copy()
        nonfinite["lidar2img"].flat[0] = np.nan
        cases.append((nonfinite, "finite"))
        not_array = dict(values)
        not_array["has_prev_bev"] = [0]
        cases.append((not_array, "NumPy ndarray"))

        for case, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValidationError, message):
                    validate_step_pipeline_inputs(self.graph, case)  # type: ignore[arg-type]

        invalid_flag = dict(values)
        invalid_flag["has_prev_bev"] = np.asarray([2], dtype=np.int32)
        with self.assertRaisesRegex(ValidationError, "0 or 1"):
            validate_step_pipeline_inputs(self.graph, invalid_flag)

        for index in (0, 1, 2, 17):
            invalid_motion = dict(values)
            invalid_motion["can_bus_delta"] = values["can_bus_delta"].copy()
            invalid_motion["can_bus_delta"][0, index] = np.float32(0.25)
            with self.subTest(cold_motion_index=index):
                with self.assertRaisesRegex(ValidationError, "cold rows"):
                    validate_step_pipeline_inputs(self.graph, invalid_motion)


if __name__ == "__main__":
    unittest.main()
