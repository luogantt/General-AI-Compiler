"""CPU mocks validate scope/edge constraints; these are not GPU accuracy tests."""

import json
from types import SimpleNamespace
import unittest

from bevformer_trt.precision_policy import (
    apply_precision_policy, classify_scope, constrain_plugin_float_edges,
    policy_definition, policy_fingerprint,
)


TRT = SimpleNamespace(
    float16="float16", float32="float32", int8="int8", int32="int32",
    LayerType=SimpleNamespace(**{name: name for name in (
        "CONVOLUTION", "MATRIX_MULTIPLY", "ACTIVATION", "POOLING", "ELEMENTWISE",
        "REDUCE", "SOFTMAX", "NORMALIZATION", "SHAPE", "SHUFFLE", "IDENTITY", "PLUGIN_V2", "PLUGIN_V3",
    )}),
    BuilderFlag=SimpleNamespace(**{name: name for name in ("INT8", "FP16", "OBEY_PRECISION_CONSTRAINTS", "TF32")}),
)


class Tensor:
    def __init__(self, dtype="float32", shape_tensor=False):
        self.dtype = dtype
        self.is_shape_tensor = shape_tensor


class Layer:
    def __init__(self, name, kind="CONVOLUTION", inputs=None, outputs=None):
        self.name, self.type = name, kind
        self.inputs = inputs if inputs is not None else [Tensor()]
        self.outputs = outputs if outputs is not None else [Tensor()]
        self.output_types = {}
        self._precision = None

    @property
    def precision(self):
        return self._precision

    @precision.setter
    def precision(self, value):
        if self.type.startswith("PLUGIN"):
            raise AssertionError("Mixed-type plugins must not receive layer.precision")
        self._precision = value

    @property
    def num_inputs(self):
        return len(self.inputs)

    @property
    def num_outputs(self):
        return len(self.outputs)

    def get_input(self, index):
        return self.inputs[index]

    def get_output(self, index):
        return self.outputs[index]

    def set_input(self, index, tensor):
        self.inputs[index] = tensor

    def set_output_type(self, index, dtype):
        self.output_types[index] = dtype


class Network:
    def __init__(self, *layers):
        self.layers = list(layers)

    @property
    def num_layers(self):
        return len(self.layers)

    def get_layer(self, index):
        return self.layers[index]

    def add_identity(self, tensor):
        layer = Layer("", "IDENTITY", inputs=[tensor], outputs=[Tensor(tensor.dtype)])
        self.layers.append(layer)
        return layer


class Config:
    def __init__(self):
        self.flags = {"TF32"}

    def set_flag(self, flag):
        self.flags.add(flag)

    def clear_flag(self, flag):
        self.flags.discard(flag)


def apply(*layers, plugins_fp32=False):
    network, config = Network(*layers), Config()
    report = apply_precision_policy(network, config, TRT, plugins_fp32=plugins_fp32)
    return network, config, report


class PrecisionPolicyTests(unittest.TestCase):
    def test_camera_repeats_and_downsample_are_int8(self):
        layers = [Layer(f"/img_backbone{suffix}/layer{block}/layer{block}.0/downsample/Conv")
                  for suffix in ("", "_1", "_2", "_3", "_4", "_5") for block in range(1, 5)]
        _, config, report = apply(*layers)
        self.assertTrue(all(layer.precision == TRT.int8 for layer in layers))
        self.assertTrue(all(layer.output_types == {0: TRT.float16} for layer in layers))
        self.assertEqual(config.flags, {"INT8", "FP16", "OBEY_PRECISION_CONSTRAINTS"})
        self.assertEqual(report["constraint_counts"], {"int8": 24})

    def test_stem_fpn_head_and_backbone_activations_fp16(self):
        layers = [
            Layer("/img_backbone_3/conv1/Conv"),
            Layer("/img_backbone/bn1/scale", "ELEMENTWISE"),
            Layer("/img_backbone/relu/Relu", "ACTIVATION"),
            Layer("/img_backbone/maxpool/MaxPool", "POOLING"),
            Layer("/img_neck/fpn_convs.0/Conv"),
            Layer("/pts_bbox_head/transformer/MatMul", "MATRIX_MULTIPLY"),
            Layer("/pts_bbox_head/Softmax", "SOFTMAX"),
            Layer("/pts_bbox_head/LayerNorm", "NORMALIZATION"),
            Layer("/img_backbone/layer2/relu/Relu", "ACTIVATION"),
        ]
        apply(*layers)
        self.assertTrue(all(layer.precision == TRT.float16 for layer in layers))

    def test_offset_mask_predictors_remain_fp16(self):
        offsets = [Layer(f"/img_backbone_2/layer3/layer3.0/conv2/{name}/Conv")
                   for name in ("conv_offset", "conv_offset_5", "conv_offset_mask", "offset_conv", "conv_mask")]
        ordinary = Layer("/img_backbone_2/layer3/layer3.0/conv1/Conv")
        _, _, report = apply(*offsets, ordinary)
        self.assertTrue(all(layer.precision == TRT.float16 for layer in offsets))
        self.assertEqual(ordinary.precision, TRT.int8)
        self.assertEqual(len(report["exceptions"]), len(offsets))

    def test_integer_and_shape_layers_untouched(self):
        layers = [
            Layer("/img_backbone/layer2/shape_add", "ELEMENTWISE", outputs=[Tensor("int32", True)]),
            Layer("/pts_bbox_head/Shape", "SHAPE", outputs=[Tensor("int32", True)]),
            Layer("/pts_bbox_head/shape_float", "ELEMENTWISE", outputs=[Tensor("float32", True)]),
            Layer("/pts_bbox_head/reshape", "SHUFFLE"),
        ]
        apply(*layers)
        self.assertTrue(all(layer.precision is None and not layer.output_types for layer in layers))

    def test_msda_only_floating_edges_constrained(self):
        inputs = [Tensor(), Tensor("int32"), Tensor("int32"), Tensor(), Tensor()]
        layer = Layer("/pts_bbox_head/MSDeformAttn", "PLUGIN_V3", inputs=inputs.copy())
        network, _, report = apply(layer)
        self.assertEqual(network.num_layers, 4)
        self.assertIs(layer.get_input(1), inputs[1])
        self.assertIs(layer.get_input(2), inputs[2])
        self.assertIsNone(layer.precision)
        self.assertEqual(layer.output_types, {0: TRT.float16})
        self.assertEqual(report["layers"][0]["float_input_indices"], [0, 3, 4])
        self.assertTrue(all(identity.output_types == {0: TRT.float16} for identity in network.layers[1:]))

    def test_dcn_float_exception_and_fp32_override(self):
        dcn = Layer("/img_backbone/layer3/conv2/MMCVModulatedDeformConv2d", "PLUGIN_V3",
                    inputs=[Tensor() for _ in range(4)])
        msda = Layer("/pts_bbox_head/MSDeformAttn", "PLUGIN_V3",
                     inputs=[Tensor(), Tensor("int32"), Tensor("int32"), Tensor(), Tensor()])
        _, _, report = apply(dcn, msda, plugins_fp32=True)
        self.assertIsNone(dcn.precision)
        self.assertEqual(dcn.output_types, {0: TRT.float32})
        self.assertEqual(msda.output_types, {0: TRT.float32})
        self.assertTrue(any("no INT8" in item["reason"] for item in report["exceptions"]))
        self.assertEqual(sum("override" in item["reason"] for item in report["exceptions"]), 2)

    def test_shared_plugin_helper_keeps_integer_output(self):
        layer = Layer("plugin", "PLUGIN_V3", outputs=[Tensor(), Tensor("int32")])
        record = constrain_plugin_float_edges(Network(layer), layer, TRT, precision="fp32")
        self.assertEqual(layer.output_types, {0: TRT.float32})
        self.assertEqual(record["float_output_indices"], [0])

    def test_unknown_float_plugin_requires_rule(self):
        with self.assertRaisesRegex(ValueError, "explicit precision rule"):
            apply(Layer("/custom/UnknownPlugin", "PLUGIN_V3"))

    def test_scope_boundaries(self):
        self.assertEqual(classify_scope("/img_backbone_5/layer1/layer1.0/Conv"), "backbone_body")
        self.assertEqual(classify_scope("img_backbone.layer4.block.conv"), "backbone_body")
        self.assertEqual(classify_scope("/img_backbone/layer10/Conv"), "backbone_other")
        self.assertEqual(classify_scope("/other_img_backbone/layer1/Conv"), "head_or_other")
        self.assertEqual(classify_scope("/img_backbone_1/stem/Conv"), "backbone_stem")

    def test_policy_fingerprint_and_unverified_report(self):
        first = policy_fingerprint()
        self.assertEqual(first, policy_fingerprint())
        self.assertNotEqual(first, policy_fingerprint(plugins_fp32=True))
        self.assertEqual(len(first), 64)
        _, _, report = apply(Layer("/img_backbone/layer1/Conv"))
        self.assertEqual(report["fingerprint_sha256"], first)
        self.assertEqual(report["rules"], policy_definition())
        self.assertEqual(report["validation"]["actual_engine_precision"], "not_checked")
        self.assertEqual(report["validation"]["performance"], "not_measured")
        json.dumps(report, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
