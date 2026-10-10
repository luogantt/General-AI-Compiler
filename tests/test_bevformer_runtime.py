from __future__ import annotations

from threading import Event, Thread
from pathlib import Path
import unittest

import numpy as np

from gac.bevformer_runtime import BevFormerRuntime, BevFormerStepContract
from gac.errors import BackendError, ValidationError
from gac.frontends.bevformer_export import load_bevformer_manifest


ROOT = Path(__file__).resolve().parents[1]


class FakeSession:
    def __init__(self, contract: BevFormerStepContract) -> None:
        self.contract = contract
        self.calls = []
        self.fail = False
        self.bad_output = False
        self.closed = False

    def run(self, inputs):
        self.calls.append({name: value.copy() for name, value in inputs.items()})
        if self.fail:
            raise BackendError("injected failure")
        c = self.contract
        next_bev = inputs["prev_bev"] + np.asarray(1, dtype=c.state_dtype)
        outputs = {
            "cls_logits": np.zeros(c.cls_shape, np.float32),
            "bbox_regression": np.zeros(c.bbox_shape, np.float32),
            "next_bev": np.ascontiguousarray(next_bev),
        }
        if self.bad_output:
            outputs["next_bev"] = np.full(c.prev_bev_shape, np.nan, dtype=c.state_dtype)
        return outputs

    def close(self):
        self.closed = True


def inputs(contract: BevFormerStepContract, *, position=(0, 0, 0), angle=0):
    images = np.zeros(contract.image_shape, dtype=contract.image_dtype)
    lidar2img = np.zeros(contract.lidar2img_shape, np.float32)
    can_bus = np.zeros(contract.can_bus_shape, np.float32)
    can_bus[0, :3] = position
    can_bus[0, -1] = angle
    return images, lidar2img, can_bus


class BevFormerRuntimeTests(unittest.TestCase):
    def setUp(self):
        # Small dimensions preserve the production ABI while keeping tests cheap.
        self.contract = BevFormerStepContract(
            cameras=2,
            image_h=4,
            image_w=6,
            bev_queries=5,
            embed_dim=8,
            object_queries=3,
            classes=2,
            bbox_dims=4,
        )
        self.session = FakeSession(self.contract)
        self.runtime = BevFormerRuntime(self.session, contract=self.contract)

    def test_cold_warm_scene_reset_contract(self):
        first_inputs = list(inputs(self.contract, position=(10, 20, 2), angle=30))
        first_inputs[2][0, 3:-1] = np.arange(
            1, self.contract.can_bus_dims - 3, dtype=np.float32
        )
        first = self.runtime.step(*first_inputs, "scene-a")
        self.assertTrue(first.cold_start)
        self.assertEqual(first.frame_index, 1)
        np.testing.assert_array_equal(self.session.calls[0]["has_prev_bev"], [0])
        np.testing.assert_array_equal(
            self.session.calls[0]["can_bus_delta"][0, :3], 0
        )
        self.assertEqual(self.session.calls[0]["can_bus_delta"][0, -1], 0)
        np.testing.assert_array_equal(
            self.session.calls[0]["can_bus_delta"][0, 3:-1],
            first_inputs[2][0, 3:-1],
        )
        np.testing.assert_array_equal(first.next_bev, np.ones(self.contract.prev_bev_shape))

        second = self.runtime.step(*inputs(self.contract, position=(13, 18, 4), angle=37), "scene-a")
        self.assertFalse(second.cold_start)
        self.assertEqual(second.frame_index, 2)
        np.testing.assert_array_equal(self.session.calls[1]["has_prev_bev"], [1])
        np.testing.assert_array_equal(self.session.calls[1]["can_bus_delta"][0, :3], [3, -2, 2])
        self.assertEqual(self.session.calls[1]["can_bus_delta"][0, -1], 7)
        np.testing.assert_array_equal(self.session.calls[1]["prev_bev"], first.next_bev)

        changed = self.runtime.step(*inputs(self.contract, position=(99, 1, 1), angle=180), "scene-b")
        self.assertTrue(changed.cold_start)
        self.assertEqual(changed.frame_index, 1)
        np.testing.assert_array_equal(self.session.calls[2]["has_prev_bev"], [0])
        np.testing.assert_array_equal(
            self.session.calls[2]["can_bus_delta"][0, :3], 0
        )
        self.assertEqual(self.session.calls[2]["can_bus_delta"][0, -1], 0)
        np.testing.assert_array_equal(self.session.calls[2]["prev_bev"], 0)

    def test_failure_and_invalid_output_do_not_commit_state(self):
        self.runtime.step(*inputs(self.contract, position=(1, 2, 3), angle=4), "scene")
        before = self.runtime.state
        self.session.fail = True
        with self.assertRaisesRegex(BackendError, "injected"):
            self.runtime.step(*inputs(self.contract, position=(9, 9, 9), angle=9), "scene")
        after = self.runtime.state
        np.testing.assert_array_equal(after.prev_bev, before.prev_bev)
        np.testing.assert_array_equal(after.last_position, before.last_position)
        self.assertEqual(after.frame_index, before.frame_index)

        self.session.fail = False
        self.session.bad_output = True
        with self.assertRaises(ValidationError):
            self.runtime.step(*inputs(self.contract, position=(8, 8, 8), angle=8), "scene")
        self.assertEqual(self.runtime.state.frame_index, before.frame_index)

    def test_state_is_a_copy_and_inputs_are_strict(self):
        snapshot = self.runtime.state
        snapshot.prev_bev.fill(99)
        np.testing.assert_array_equal(self.runtime.state.prev_bev, 0)

        images, lidar2img, can_bus = inputs(self.contract)
        wrong_image_dtype = np.float32 if images.dtype == np.float16 else np.float16
        with self.assertRaises(ValidationError):
            self.runtime.step(images.astype(wrong_image_dtype), lidar2img, can_bus, "scene")
        with self.assertRaises(ValidationError):
            self.runtime.step(images, lidar2img[:, :1], can_bus, "scene")
        can_bus[0, 0] = np.nan
        with self.assertRaises(ValidationError):
            self.runtime.step(images, lidar2img, can_bus, "scene")

    def test_concurrent_call_is_rejected(self):
        entered = Event()
        release = Event()
        original_run = self.session.run

        def blocking_run(call_inputs):
            entered.set()
            release.wait(2)
            return original_run(call_inputs)

        self.session.run = blocking_run
        args = (*inputs(self.contract), "scene")
        thread = Thread(target=lambda: self.runtime.step(*args))
        thread.start()
        self.assertTrue(entered.wait(1))
        with self.assertRaisesRegex(BackendError, "concurrent"):
            self.runtime.step(*args)
        release.set()
        thread.join(2)
        self.assertFalse(thread.is_alive())

    def test_reset_close_and_context_manager(self):
        self.runtime.step(*inputs(self.contract), "scene")
        self.runtime.reset("next")
        state = self.runtime.state
        self.assertFalse(state.valid)
        self.assertEqual(state.scene_token, "next")
        self.assertEqual(state.frame_index, 0)

        self.runtime.close()
        self.assertTrue(self.session.closed)
        self.runtime.close()
        with self.assertRaises(BackendError):
            self.runtime.step(*inputs(self.contract), "scene")

    def test_checked_in_manifest_and_runtime_contract_are_identical(self):
        manifest = load_bevformer_manifest(ROOT / "examples" / "bevformer_tiny_contract.json")
        contract = BevFormerStepContract.from_manifest(manifest)
        self.assertEqual(contract, BevFormerStepContract())
        self.assertEqual(contract.image_dtype, "float16")
        self.assertEqual(contract.state_dtype, "float16")


if __name__ == "__main__":
    unittest.main()
