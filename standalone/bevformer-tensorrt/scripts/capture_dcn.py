#!/usr/bin/env python3
r"""Capture one official DCNv2 plugin's FP16 inputs and output from full ONNX.

Build with the original complete ONNX and its calibrated range binding:
  python scripts/capture_dcn.py --onnx models/adapted.onnx --images images.npy \
    --node-substring /layer3/layer3.0/conv2/MMCVModulatedDeformConv2d \
    --precision mixed-int8 --ranges ranges.json \
    --output build/dcn-capture-build --build-only
Run in a fresh process to avoid builder/PyTorch memory overlap:
  python scripts/capture_dcn.py --engine build/dcn-capture-build/capture.engine \
    --images images.npy --output build/dcn-capture-run

The NPZ contains input, offset, mask, weight, optional bias, and expected.
Only static NCHW and the adapted model's symmetric stride/padding/dilation
conventions are accepted. Capturing boundaries can change TensorRT tactics.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import capture_msda as common


PLUGIN = "ModulatedDeformConv2d"
INPUT_KEYS = ("input", "offset", "mask", "weight", "bias")


def validate_attributes(attributes):
    """Keep raw plugin [W,H] fields; adapted BEVFormer currently uses symmetric pairs."""
    required = {"stride", "padding", "dilation", "group", "deformable_group"}
    if set(attributes) != required:
        raise ValueError(f"DCNv2 attributes require exactly {sorted(required)}")
    result = {}
    for name in ("stride", "padding", "dilation"):
        values = attributes[name]
        minimum = 0 if name == "padding" else 1
        if (not isinstance(values, (tuple, list)) or len(values) != 2
                or any(type(value) is not int or value < minimum for value in values)):
            raise ValueError(f"DCNv2 {name} must be two integers >= {minimum}")
        if values[0] != values[1]:
            raise ValueError(f"Asymmetric DCNv2 {name} is outside the verified adapted-model convention")
        result[name] = list(values)
    for name in ("group", "deformable_group"):
        value = attributes[name]
        if type(value) is not int or value <= 0:
            raise ValueError(f"DCNv2 {name} must be a positive integer")
        result[name] = value
    return result


def select_node(path, substring):
    import onnx

    common.engine_tools.require_embedded_weights(path)
    model = onnx.load(path, load_external_data=False)
    available = [node for node in model.graph.node if node.op_type == PLUGIN]
    selected = [node for node in available if substring in node.name]
    if len(selected) != 1:
        raise ValueError(f"Expected one official DCNv2 matching {substring!r}; found {len(selected)}. "
                         f"Available nodes: {[node.name for node in available]}")
    node = selected[0]
    if (len(node.input) not in (4, 5) or len(node.output) != 1
            or any(not name for name in (*node.input, *node.output))):
        raise ValueError("DCNv2 requires four/five nonempty inputs and one output")
    attrs = {attribute.name: onnx.helper.get_attribute_value(attribute) for attribute in node.attribute}
    if len(attrs) != len(node.attribute):
        raise ValueError("Duplicate DCNv2 attributes")
    version = attrs.pop("plugin_version", b"1")
    namespace = attrs.pop("plugin_namespace", b"")
    version = version.decode() if isinstance(version, bytes) else str(version)
    namespace = namespace.decode() if isinstance(namespace, bytes) else str(namespace)
    if version != "2" or namespace:
        raise ValueError("Capture requires NVIDIA DCNv2 plugin version 2, empty namespace")
    attrs = validate_attributes(attrs)
    return {"name": node.name, "op_type": node.op_type, "domain": node.domain,
            "inputs": list(node.input), "outputs": list(node.output), "has_bias": len(node.input) == 5,
            "plugin_version": version, "plugin_namespace": namespace, "attributes": attrs,
            "attribute_pair_order": "[width,height]; symmetric pairs only", "layout": "NCHW"}


def validate_arrays(arrays, np, selected):
    keys = DCNCaptureSpec.keys(selected)
    if set(arrays) != set(keys):
        raise ValueError("DCNv2 archive keys disagree with the selected plugin ABI")
    attrs = validate_attributes(selected["attributes"])
    if any(array.dtype != np.float16 for array in arrays.values()):
        raise ValueError("DCNv2 capture requires FP16 for all inputs and expected output")
    if any(arrays[key].ndim != 4 for key in ("input", "offset", "mask", "weight", "expected")):
        raise ValueError("DCNv2 input, offset, mask, weight and expected must have rank four")
    n, ci, hi, wi = arrays["input"].shape
    co, weight_ci, kh, kw = arrays["weight"].shape
    group, deformable_group = attrs["group"], attrs["deformable_group"]
    if (min(n, ci, hi, wi, co, weight_ci, kh, kw) <= 0 or ci % group or co % group
            or ci % deformable_group or weight_ci != ci // group):
        raise ValueError("DCNv2 grouped channel dimensions are inconsistent or empty")
    sw, sh = attrs["stride"]
    pw, ph = attrs["padding"]
    dw, dh = attrs["dilation"]
    ho = (hi + 2 * ph - dh * (kh - 1) - 1) // sh + 1
    wo = (wi + 2 * pw - dw * (kw - 1) - 1) // sw + 1
    expected_shapes = {"offset": (n, 2 * deformable_group * kh * kw, ho, wo),
                       "mask": (n, deformable_group * kh * kw, ho, wo),
                       "expected": (n, co, ho, wo)}
    if "bias" in arrays:
        expected_shapes["bias"] = (co,)
    if min(ho, wo) <= 0 or any(arrays[key].shape != shape for key, shape in expected_shapes.items()):
        raise ValueError(f"DCNv2 captured shapes disagree with convolution metadata: {expected_shapes}")
    finite = {key: bool(np.isfinite(array).all()) for key, array in arrays.items()}
    if not all(finite.values()):
        raise ValueError(f"DCNv2 captured nonfinite tensors: {finite}")
    return {"status": "passed", "layout": "NCHW", "dtype": "float16", "finite": finite,
            "attributes": attrs, "dimensions": {"N": n, "Cin": ci, "Cout": co, "H": hi, "W": wi,
                                                  "Kh": kh, "Kw": kw, "Ho": ho, "Wo": wo}}


class DCNCaptureSpec:
    schema = "gac.dcn.real_capture.v1"
    prefix = "gac_dcn_capture"
    operation = "dcn"
    integer_keys = ()
    default_node_substring = "/layer3/layer3.0/conv2/MMCVModulatedDeformConv2d"
    precisions = ("fp16", "mixed-int8")
    description = __doc__
    expected_description = "official NVIDIA ModulatedDeformConv2d v2 output from capture engine"
    impact = ("Parsed the original complete ONNX and applied its precision/range policy before exposing "
              "the selected DCNv2 inputs/output as constrained FP16 Identity boundaries. Removed original "
              "output marks; TensorRT may prune downstream work and choose different tactics/reformats. "
              "These real-image activations belong to the capture engine, not necessarily the original engine.")
    select_node = staticmethod(select_node)
    validate_arrays = staticmethod(validate_arrays)

    @staticmethod
    def keys(selected):
        return (*INPUT_KEYS[:len(selected["inputs"])], "expected")


def main(argv=None):
    return common.main(argv, capture=DCNCaptureSpec())


if __name__ == "__main__":
    sys.exit(main())
