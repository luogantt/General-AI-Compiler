"""DCNv2 real-capture selection and NCHW metadata contracts, on CPU."""

from copy import deepcopy
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np
import onnx
from onnx import helper


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("dcn_capture_test", ROOT / "scripts" / "capture_dcn.py")
CAPTURE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CAPTURE)


def attributes():
    return dict(stride=[2, 2], padding=[1, 1], dilation=[1, 1], group=2, deformable_group=2)


def node(name="/img_backbone/layer3/layer3.0/conv2/MMCVModulatedDeformConv2d", bias=False, version="2"):
    inputs = ["input", "offset", "mask", "weight"] + (["bias"] if bias else [])
    return helper.make_node("ModulatedDeformConv2d", inputs, ["expected"], name=name,
                            domain="trt.plugins", plugin_version=version, plugin_namespace="", **attributes())


def fixture(bias=False):
    selected = {"inputs": ["input", "offset", "mask", "weight"] + (["bias"] if bias else []),
                "attributes": attributes()}
    shapes = {"input": (2, 4, 5, 7), "offset": (2, 36, 3, 4), "mask": (2, 18, 3, 4),
              "weight": (6, 2, 3, 3), "expected": (2, 6, 3, 4)}
    if bias:
        shapes["bias"] = (6,)
    return {key: np.zeros(shape, dtype=np.float16) for key, shape in shapes.items()}, selected


class DCNCaptureTests(unittest.TestCase):
    def write_model(self, folder, nodes):
        path = Path(folder) / "adapted.onnx"
        graph = helper.make_graph(nodes, "capture_selection", [], [])
        onnx.save(helper.make_model(graph), path)
        return path

    def test_select_unique_node_preserves_stride_groups_and_bias(self):
        with tempfile.TemporaryDirectory() as folder:
            for bias in (False, True):
                path = self.write_model(folder, [node(bias=bias), node(name="/another/dcn")])
                selected = CAPTURE.select_node(path, CAPTURE.DCNCaptureSpec.default_node_substring)
                self.assertEqual(selected["attributes"], attributes())
                self.assertEqual(selected["has_bias"], bias)
                self.assertEqual(selected["plugin_version"], "2")
                self.assertEqual(selected["layout"], "NCHW")
                self.assertEqual(CAPTURE.DCNCaptureSpec.keys(selected),
                                 ("input", "offset", "mask", "weight", *(["bias"] if bias else []), "expected"))

    def test_selection_rejects_ambiguous_missing_wrong_version_and_empty_input(self):
        with tempfile.TemporaryDirectory() as folder:
            path = self.write_model(folder, [node(), node(name="/other/MMCVModulatedDeformConv2d")])
            for substring in ("MMCVModulatedDeformConv2d", "missing"):
                with self.assertRaisesRegex(ValueError, "Expected one official DCNv2"):
                    CAPTURE.select_node(path, substring)
            path = self.write_model(folder, [node(version="1")])
            with self.assertRaisesRegex(ValueError, "version 2"):
                CAPTURE.select_node(path, "layer3.0")
            invalid = node(bias=True)
            invalid.input[-1] = ""
            path = self.write_model(folder, [invalid])
            with self.assertRaisesRegex(ValueError, "nonempty inputs"):
                CAPTURE.select_node(path, "layer3.0")

    def test_metadata_and_groups_match_nchw_convolution(self):
        for bias in (False, True):
            arrays, selected = fixture(bias)
            checked = CAPTURE.validate_arrays(arrays, np, selected)
            self.assertEqual(checked["status"], "passed")
            self.assertEqual(checked["dimensions"]["Ho"], 3)
            self.assertEqual(checked["dimensions"]["Wo"], 4)
            self.assertEqual(checked["attributes"]["stride"], [2, 2])
            self.assertTrue(all(checked["finite"].values()))

    def test_rejects_invalid_fields_channels_offsets_dtype_and_nonfinite(self):
        for changed in ({"stride": [1, 2]}, {"padding": [-1, -1]}, {"group": True},
                        {"dilation": [1.0, 1.0]}, {"extra": 1}):
            with self.assertRaises(ValueError):
                CAPTURE.validate_attributes({**attributes(), **changed})
        arrays, selected = fixture()
        invalid_arrays = []
        wrong_offset = arrays.copy()
        wrong_offset["offset"] = np.zeros((2, 18, 3, 4), dtype=np.float16)
        invalid_arrays.append(wrong_offset)
        wrong_dtype = arrays.copy()
        wrong_dtype["weight"] = arrays["weight"].astype(np.float32)
        invalid_arrays.append(wrong_dtype)
        nonfinite = deepcopy(arrays)
        nonfinite["expected"].flat[0] = np.nan
        invalid_arrays.append(nonfinite)
        for invalid in invalid_arrays:
            with self.assertRaises(ValueError):
                CAPTURE.validate_arrays(invalid, np, selected)
        wrong_groups = deepcopy(selected)
        wrong_groups["attributes"]["deformable_group"] = 3
        with self.assertRaisesRegex(ValueError, "grouped channel dimensions"):
            CAPTURE.validate_arrays(arrays, np, wrong_groups)

    def test_shared_msda_defaults_keep_existing_schema_and_keys(self):
        spec = CAPTURE.common.MSDACaptureSpec()
        self.assertEqual(spec.schema, "gac.msda.real_capture.v1")
        self.assertEqual(spec.keys({}), ("value", "shapes", "starts", "locations", "weights", "expected"))
        self.assertEqual(spec.integer_keys, ("shapes", "starts"))
        self.assertEqual(spec.default_node_substring, "encoder/layers.0/attentions.1")
        self.assertEqual(spec.description, CAPTURE.common.__doc__)


if __name__ == "__main__":
    unittest.main()
