from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "tools/galaxeavla/profile_g05_modules.py"
SPEC = importlib.util.spec_from_file_location("profile_g05_modules", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class Module:
    def forward(self, value):
        return value

    def embed(self, value):
        return value

    def decode(self, value):
        return value

    def encode_time(self, value):
        return value


def policy():
    return SimpleNamespace(model=SimpleNamespace(vision_tower=Module(), vlm=Module(), action_expert=Module()))


class ModuleProfilerTests(unittest.TestCase):
    def test_records_synchronized_calls_without_changing_outputs_or_method_ownership(self):
        model = policy()
        events = []
        times = iter((1., 1.003, 2., 2.005))
        def clock():
            events.append("clock")
            return next(times)
        profiler = MODULE.ModuleProfiler(model, lambda: events.append("sync"), clock=clock)
        token = object()
        with profiler:
            with profiler.phase("request_0"):
                self.assertIs(model.model.vision_tower.forward(token), token)
                self.assertIs(model.model.action_expert.embed(token), token)
        self.assertEqual(events, ["sync", "clock", "sync", "clock"] * 2)
        self.assertNotIn("forward", vars(model.model.vision_tower))
        self.assertNotIn("embed", vars(model.model.action_expert))
        summary = profiler.summary()
        self.assertFalse(summary["attached"])
        self.assertEqual(summary["modules"]["vision_tower.forward"]["calls"], 1)
        self.assertAlmostEqual(summary["modules"]["action_expert.embed"]["total_ms"], 5)
        self.assertEqual(summary["modules"]["action_expert.forward"]["calls"], 0)
        self.assertTrue(all(row["phase"] == "request_0" for row in summary["calls"]))

    def test_existing_compiled_forward_restored_and_exception_reported(self):
        model = policy()
        def compiled(value):
            raise ValueError("original failure")
        model.model.action_expert.forward = compiled
        profiler = MODULE.ModuleProfiler(model, lambda: None, clock=lambda: 1.)
        with self.assertRaisesRegex(ValueError, "original failure"):
            with profiler:
                model.model.action_expert.forward(None)
        self.assertIs(model.model.action_expert.forward, compiled)
        self.assertEqual(profiler.summary()["modules"]["action_expert.forward"]["errors"], 1)
        profiler.close()  # Idempotent.
        self.assertIs(model.model.action_expert.forward, compiled)

    def test_disabled_profiler_does_not_wrap_or_synchronize(self):
        model = policy()
        def unwanted():
            raise AssertionError("disabled profiler synchronized")
        with MODULE.ModuleProfiler(model, unwanted, enabled=False) as profiler:
            self.assertNotIn("forward", vars(model.model.vision_tower))
            model.model.vision_tower.forward(None)
        self.assertEqual(profiler.summary()["calls"], [])

    def test_invalid_policy_or_double_installation_leaves_original_methods_intact(self):
        model = policy()
        model.model.action_expert.decode = None
        with self.assertRaisesRegex(TypeError, "decode"):
            MODULE.ModuleProfiler(model, lambda: None).attach()
        self.assertNotIn("forward", vars(model.model.vision_tower))
        model.model.action_expert.decode = lambda value: value
        with MODULE.ModuleProfiler(model, lambda: None):
            installed = model.model.vision_tower.forward
            with self.assertRaisesRegex(RuntimeError, "already attached"):
                MODULE.ModuleProfiler(model, lambda: None).attach()
            self.assertIs(model.model.vision_tower.forward, installed)
        self.assertNotIn("forward", vars(model.model.vision_tower))

    def test_restore_does_not_overwrite_later_external_replacement(self):
        model = policy()
        replacement = lambda value: ("new", value)
        with MODULE.ModuleProfiler(model, lambda: None):
            model.model.vlm.forward = replacement
        self.assertIs(model.model.vlm.forward, replacement)


if __name__ == "__main__":
    unittest.main()
