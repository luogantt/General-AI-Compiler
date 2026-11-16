"""CPU-only tests for removal of dead action preprocessing, using no torch."""
from __future__ import annotations

from copy import deepcopy
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch


PATH = Path(__file__).resolve().parents[1] / "tools/galaxeavla/g05_context_only.py"
SPEC = importlib.util.spec_from_file_location("g05_context_only", PATH)
HELPER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = HELPER
SPEC.loader.exec_module(HELPER)


def make_source():
    def process(self, data, **kwargs):
        return data

    classes = {name: type(name, (), {"process": process}) for name in HELPER._SAFE_PROCESSOR_NAMES}

    class InputPreprocessor:
        def __init__(self):
            self.modality_processors = {
                "text": classes["BuiltinTextProcessor"](),
                "image": classes["BuiltinQwen35ImageProcessor"](),
                "proprio": classes["BuiltinProprioMLPProcessor"](),
            }
            self.codec_calls = 0
            self.calls = []

        def _parse_template(self, template):
            return template.split()

        def _parse_control_tokens(self, tokens, control_flag):
            prefix = tokens[:tokens.index("EOC")] if "EOC" in tokens else tokens
            return [SimpleNamespace(type="dynamic", sample_key=token.split(":")[0],
                                    processor_key=token.split(":")[1]) for token in prefix]

        def _prepare_segments(self, samples, template=None, control_flag=None):
            return [self._parse_control_tokens(self._parse_template(template or s["template"]), control_flag)
                    for s in samples]

        def _get_processor(self, key):
            return self.modality_processors[key]

        def _preprocess_single_sample(self, sample):
            # Synthetic token contents encode all current dynamic inputs. The
            # dead action suffix is deliberately not part of the context output.
            return sample["command"], tuple(sample["proprio"]), tuple(sample["image0"])

        def preprocess(self, samples, **kwargs):
            self.codec_calls += sum("action" in sample for sample in samples)
            return tuple(self._preprocess_single_sample(sample) for sample in samples)

        def encode_inference(self, samples, device, mode="fm", training=False, **kwargs):
            private = deepcopy(samples)
            self.calls.append((private, device, mode, training, kwargs))
            return self.preprocess(private)

    return SimpleNamespace(InputPreprocessor=InputPreprocessor, **classes)


class ContextOnlyTests(unittest.TestCase):
    def setUp(self):
        self.source = make_source()
        self.processor = self.source.InputPreprocessor()
        self.policy = SimpleNamespace(processor=self.processor, training=False,
                                      continuous_action=True, discrete_action=False, predict_cot=False)
        self.import_patch = patch.object(HELPER.importlib, "import_module", return_value=self.source)
        self.import_patch.start()
        self.addCleanup(self.import_patch.stop)

    def sample(self):
        return {"command": "open drawer", "proprio": [1.0, 2.0], "image0": [256, 256],
                "action": {"value": [7.0, 8.0]},
                "template": "command:text proprio:proprio image0:image EOC action:action"}

    def test_same_outputs_without_codec_work_and_current_dynamic_inputs_each_call(self):
        samples = [self.sample()]
        original_snapshot = deepcopy(samples)
        expected = self.processor.encode_inference(samples, "cpu", mode="ar")
        self.assertEqual(self.processor.codec_calls, 1)
        handle = HELPER.install_context_only_optimization(self.policy)
        actual = self.processor.encode_inference(samples, "cpu", mode="ar")
        self.assertEqual(actual, expected)
        self.assertEqual(self.processor.codec_calls, 1)
        self.assertEqual(samples, original_snapshot)
        changed = deepcopy(samples)
        changed[0]["command"] = "close middle drawer"
        changed[0]["proprio"][0] = -3.5
        changed[0]["image0"] = [192, 320]
        changed_result = self.processor.encode_inference(changed, "npu:0", mode="ar")
        self.assertNotEqual(changed_result, actual)
        self.assertEqual(changed_result[0], ("close middle drawer", (-3.5, 2.0), (192, 320)))
        self.assertIn("action", changed[0])
        self.assertNotIn("action", self.processor.calls[-1][0][0])
        self.assertEqual(self.processor.calls[-1][1], "npu:0")
        self.assertEqual(handle.metadata["samples_with_action_removed"], 2)
        self.assertFalse(handle.metadata["cached_model_outputs"])

    def test_retained_action_reference_or_action_processor_is_rejected(self):
        HELPER.install_context_only_optimization(self.policy)
        for template in ("action:text EOC", "other:action EOC", "command:text action:action"):
            sample = self.sample()
            sample["template"] = template
            with self.subTest(template=template), self.assertRaisesRegex(ValueError, "references action"):
                self.processor.encode_inference([sample], "cpu", mode="ar")
        self.assertEqual(self.processor.codec_calls, 0)

    def test_template_override_is_checked_and_no_action_case_is_supported(self):
        handle = HELPER.install_context_only_optimization(self.policy)
        sample = self.sample()
        sample.pop("action")
        result = self.processor.encode_inference([sample], "cpu", mode="ar", template="command:text EOC")
        self.assertEqual(result[0][0], "open drawer")
        self.assertEqual(handle.metadata["samples_without_action"], 1)
        with self.assertRaisesRegex(ValueError, "references action"):
            self.processor.encode_inference([sample], "cpu", mode="ar", template="action:text EOC")

    def test_custom_subclass_or_replaced_processor_method_is_rejected(self):
        HELPER.install_context_only_optimization(self.policy)
        class CustomText(self.source.BuiltinTextProcessor):
            pass
        self.processor.modality_processors["text"] = CustomText()
        with self.assertRaisesRegex(TypeError, "Unsafe context modality"):
            self.processor.encode_inference([self.sample()], "cpu", mode="ar")
        known = self.source.BuiltinTextProcessor()
        known.process = lambda *args, **kwargs: None
        self.processor.modality_processors["text"] = known
        with self.assertRaisesRegex(TypeError, "Replaced process"):
            self.processor.encode_inference([self.sample()], "cpu", mode="ar")

    def test_mode_and_policy_gates_are_rechecked_on_every_request(self):
        HELPER.install_context_only_optimization(self.policy)
        with self.assertRaisesRegex(ValueError, "mode='ar'"):
            self.processor.encode_inference([self.sample()], "cpu", mode="fm")
        with self.assertRaisesRegex(ValueError, "training=False"):
            self.processor.encode_inference([self.sample()], "cpu", mode="ar", training=True)
        for name, bad in (("training", True), ("discrete_action", True),
                          ("continuous_action", False), ("predict_cot", True)):
            old = getattr(self.policy, name)
            setattr(self.policy, name, bad)
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, name):
                self.processor.encode_inference([self.sample()], "cpu", mode="ar")
            setattr(self.policy, name, old)
        self.processor._parse_template = lambda template: []
        with self.assertRaisesRegex(RuntimeError, "_parse_template"):
            self.processor.encode_inference([self.sample()], "cpu", mode="ar")

    def test_restore_is_idempotent_and_preserves_later_external_replacement(self):
        handle = HELPER.install_context_only_optimization(self.policy)
        with self.assertRaisesRegex(RuntimeError, "already installed"):
            HELPER.install_context_only_optimization(self.policy)
        handle.restore()
        handle.restore()
        self.assertNotIn("encode_inference", vars(self.processor))
        self.processor.encode_inference([self.sample()], "cpu", mode="ar")
        self.assertEqual(self.processor.codec_calls, 1)
        second = HELPER.install_context_only_optimization(self.policy)
        later = lambda *args, **kwargs: None
        self.processor.encode_inference = later
        second.restore()
        self.assertIs(self.processor.encode_inference, later)


if __name__ == "__main__":
    unittest.main()
