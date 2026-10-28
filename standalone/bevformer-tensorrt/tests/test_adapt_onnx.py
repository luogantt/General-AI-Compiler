"""CPU contract checks; GPU plugin/oracle parity is a separate validation step."""

import unittest

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from onnx.reference import ReferenceEvaluator

from bevformer_trt.adapt_onnx import AdaptationError, DCN_PLUGIN, MSDA_PLUGIN, PLUGIN_DOMAIN, adapt_model


def msda_model(*, constant_node=False, dynamic=False):
    spatial = numpy_helper.from_array(np.array([[2, 3], [1, 2]], np.int64), "spatial")
    batch, queries = ("batch", "queries") if dynamic else (2, 5)
    inputs = [
        helper.make_tensor_value_info("value", TensorProto.FLOAT, [batch, 8, 2, 4]),
        helper.make_tensor_value_info("locations", TensorProto.FLOAT, [batch, queries, 2, 2, 3, 2]),
        helper.make_tensor_value_info("weights", TensorProto.FLOAT, [batch, queries, 2, 2, 3]),
    ]
    nodes = [helper.make_node("MSDeformAttn", ["value", "spatial", "locations", "weights"], ["result"],
                              domain="gac", name="attention", mode="bilinear", padding_mode="zeros", align_corners=0)]
    if constant_node:
        nodes.insert(0, helper.make_node("Constant", [], ["spatial"], value=spatial))
    graph = helper.make_graph(nodes, "test", inputs,
                              [helper.make_tensor_value_info("result", TensorProto.FLOAT, [batch, queries, 8])],
                              initializer=[] if constant_node else [spatial])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid("gac", 1)])


def dcn_model(**overrides):
    attrs = dict(stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, deform_groups=1)
    attrs.update(overrides)
    shapes = {"X": [1, 2, 4, 4], "offset": [1, 18, 4, 4], "mask": [1, 9, 4, 4], "W": [3, 2, 3, 3]}
    graph = helper.make_graph(
        [helper.make_node("MMCVModulatedDeformConv2d", list(shapes), ["Y"], name="dcn", domain="mmcv", **attrs)],
        "test", [helper.make_tensor_value_info(n, TensorProto.FLOAT, s) for n, s in shapes.items()],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 3, 4, 4])],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid("mmcv", 1)])


def conv_alias_model(*, weight_depth=1, bias_depth=1, with_bias=True, conv_count=1,
                     identity_domain="", conv_domain=""):
    """Small actual Conv graph; alias nodes remain observable graph outputs."""
    rng = np.random.default_rng(27)
    weights = [numpy_helper.from_array(rng.normal(size=(2, 2, 1, 1)).astype(np.float32), "W")]
    nodes = [helper.make_node("Identity", ["X"], ["activation_alias"], name="activation_identity")]

    def chain(root, depth):
        value = root
        for index in range(depth):
            output = f"{root}_alias_{index}"
            nodes.append(helper.make_node("Identity", [value], [output], name=f"{root}_identity_{index}",
                                          domain=identity_domain))
            value = output
        return value

    weight_input = chain("W", weight_depth)
    if with_bias:
        weights.append(numpy_helper.from_array(rng.normal(size=(2,)).astype(np.float32), "B"))
        bias_input = chain("B", bias_depth)
    value = "activation_alias"
    outputs = []
    for index in range(conv_count):
        inputs = [value, weight_input] + ([bias_input] if with_bias else [])
        value = f"Y_{index}"
        nodes.append(helper.make_node("Conv", inputs, [value], name=f"conv_{index}", domain=conv_domain))
        outputs.append(helper.make_tensor_value_info(value, TensorProto.FLOAT, [1, 2, 3, 3]))
    # An unrelated op using the same alias must retain its original input names.
    nodes.append(helper.make_node("Add", [weight_input, weight_input], ["weight_sum"], name="other_op"))
    outputs.append(helper.make_tensor_value_info("weight_sum", TensorProto.FLOAT, [2, 2, 1, 1]))
    graph = helper.make_graph(nodes, "conv_aliases",
                              [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 2, 3, 3])],
                              outputs, initializer=weights)
    domains = {"", identity_domain, conv_domain}
    return helper.make_model(graph, opset_imports=[helper.make_opsetid(domain, 17 if domain == "" else 1)
                                                  for domain in sorted(domains)])


class AdapterTests(unittest.TestCase):
    def test_msda_initializer_and_static_reshape(self):
        source = msda_model()
        before = source.SerializeToString()
        converted, report = adapt_model(source)
        self.assertEqual(before, source.SerializeToString(), "adapter mutated source graph")
        plugin, reshape = converted.graph.node
        self.assertEqual(plugin.op_type, MSDA_PLUGIN)
        self.assertEqual(plugin.domain, PLUGIN_DOMAIN)
        self.assertEqual(len(plugin.input), 5)
        constants = {t.name: numpy_helper.to_array(t) for t in converted.graph.initializer}
        np.testing.assert_array_equal(constants[plugin.input[1]], [[2, 3], [1, 2]])
        self.assertEqual(constants[plugin.input[1]].dtype, np.int32)
        np.testing.assert_array_equal(constants[plugin.input[2]], [0, 6])
        self.assertEqual(constants[plugin.input[2]].dtype, np.int32)
        self.assertEqual(constants["spatial"].dtype, np.int64, "original constant must stay unchanged")
        self.assertEqual(reshape.op_type, "Reshape")
        self.assertEqual(list(reshape.output), ["result"])
        self.assertEqual(reshape.input[0], plugin.output[0])
        np.testing.assert_array_equal(constants[reshape.input[1]], [2, 5, 8])
        self.assertEqual(report["converted"], {"gac::MSDeformAttn": 1})
        self.assertEqual(converted.graph.output[0], source.graph.output[0])

    def test_msda_constant_node_dynamic_shape(self):
        source = msda_model(constant_node=True, dynamic=True)
        converted, report = adapt_model(source)
        self.assertEqual(converted.graph.node[0], source.graph.node[0])
        self.assertEqual(report["nodes"][0]["output_reshape"], "dynamic")
        self.assertEqual([n.op_type for n in converted.graph.node[-7:]],
                         ["Shape", "Gather", "Gather", "Gather", "Mul", "Concat", "Reshape"])
        self.assertEqual(converted.graph.node[-1].output[0], "result")
        onnx.checker.check_model(converted)

    def test_msda_dynamic_reshape_preserves_element_order(self):
        converted, _ = adapt_model(msda_model(dynamic=True))
        plugin_output = converted.graph.node[0].output[0]
        graph = helper.make_graph(
            list(converted.graph.node[1:]), "reshape_only",
            [helper.make_tensor_value_info(plugin_output, TensorProto.FLOAT, ["batch", "queries", 2, 4])],
            list(converted.graph.output), initializer=list(converted.graph.initializer),
        )
        evaluator = ReferenceEvaluator(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)]))
        for batch, queries in [(2, 5), (3, 7), (1, 0)]:
            array = np.arange(batch * queries * 8, dtype=np.float32).reshape(batch, queries, 2, 4)
            result = evaluator.run(None, {plugin_output: array})[0]
            np.testing.assert_array_equal(result, array.reshape(batch, queries, 8))

    def test_msda_malformed_spatial_shapes(self):
        for values in (np.array([2, 3]), np.array([[0, 3], [1, 2]]),
                       np.array([[2.0, 3.0], [1.0, 2.0]]),
                       np.array([[65536, 65536]], np.int64),
                       np.array([[2147483648, 1]], np.int64)):
            with self.subTest(values=values):
                model = msda_model()
                model.graph.initializer[0].CopyFrom(numpy_helper.from_array(values, "spatial"))
                with self.assertRaises(AdaptationError):
                    adapt_model(model)

    def test_msda_shape_mismatch(self):
        model = msda_model()
        model.graph.input[0].type.tensor_type.shape.dim[1].dim_value = 9
        with self.assertRaisesRegex(AdaptationError, "spatial_shapes sum"):
            adapt_model(model)
        model = msda_model()
        model.graph.input[1].type.tensor_type.shape.dim[2].dim_value = 3
        with self.assertRaisesRegex(AdaptationError, "head count"):
            adapt_model(model)

    def test_msda_unsupported_semantics(self):
        model = msda_model()
        for attribute in model.graph.node[0].attribute:
            if attribute.name == "align_corners":
                attribute.i = 1
        with self.assertRaisesRegex(AdaptationError, "align_corners"):
            adapt_model(model)
        model = msda_model()
        del model.graph.initializer[:]
        with self.assertRaisesRegex(AdaptationError, "initializer or Constant"):
            adapt_model(model)

    def test_dcn_plugin_fields(self):
        model = dcn_model(groups=2, deform_groups=2)
        converted, report = adapt_model(model)
        plugin = converted.graph.node[0]
        self.assertEqual(plugin.op_type, DCN_PLUGIN)
        attrs = {a.name: helper.get_attribute_value(a) for a in plugin.attribute}
        self.assertEqual(attrs, dict(group=2, deformable_group=2, stride=[1, 1], padding=[1, 1],
                                     dilation=[1, 1], plugin_version=b"2", plugin_namespace=b""))
        self.assertEqual(list(plugin.input), list(model.graph.node[0].input))
        self.assertFalse(report["nodes"][0]["has_bias"])

    def test_dcn_asymmetric_rejected(self):
        for key, value in (("stride", [2, 1]), ("padding", [1, 0]), ("dilation", [2, 1])):
            with self.subTest(key=key), self.assertRaisesRegex(AdaptationError, "asymmetric DCN"):
                adapt_model(dcn_model(**{key: value}))

    def test_dcn_missing_or_unknown_attributes_rejected(self):
        model = dcn_model()
        del model.graph.node[0].attribute[0]
        with self.assertRaisesRegex(AdaptationError, "missing="):
            adapt_model(model)
        with self.assertRaisesRegex(AdaptationError, "unsupported=.*unknown"):
            adapt_model(dcn_model(unknown=1))

    def test_unknown_custom_operation_rejected(self):
        model = msda_model()
        model.graph.node[0].op_type = "Unknown"
        with self.assertRaisesRegex(AdaptationError, "unsupported custom operation"):
            adapt_model(model)


class ConvWeightAliasTests(unittest.TestCase):
    def test_single_identity_weight_and_bias_are_rewritten(self):
        source = conv_alias_model()
        converted, report = adapt_model(source)
        conv = next(node for node in converted.graph.node if node.name == "conv_0")
        self.assertEqual(list(conv.input), ["activation_alias", "W", "B"])
        self.assertEqual(report["static_conv_weight_aliases"], {
            "rewritten_input_count": 2, "rewritten_conv_count": 1,
            "aliases": [
                {"node": "conv_0", "input_index": 1, "role": "weight", "previous_tensor": "W_alias_0",
                 "initializer": "W", "identity_chain": ["W_identity_0"]},
                {"node": "conv_0", "input_index": 2, "role": "bias", "previous_tensor": "B_alias_0",
                 "initializer": "B", "identity_chain": ["B_identity_0"]},
            ],
        })

    def test_shared_multilevel_chains_report_nearest_to_root_order(self):
        converted, report = adapt_model(conv_alias_model(weight_depth=3, bias_depth=2, conv_count=2))
        info = report["static_conv_weight_aliases"]
        self.assertEqual((info["rewritten_input_count"], info["rewritten_conv_count"]), (4, 2))
        aliases = {(entry["node"], entry["role"]): entry for entry in info["aliases"]}
        for index in range(2):
            weight, bias = aliases[(f"conv_{index}", "weight")], aliases[(f"conv_{index}", "bias")]
            self.assertEqual(weight["identity_chain"], ["W_identity_2", "W_identity_1", "W_identity_0"])
            self.assertEqual(bias["identity_chain"], ["B_identity_1", "B_identity_0"])
            self.assertEqual(weight["previous_tensor"], "W_alias_2")
            self.assertEqual(bias["previous_tensor"], "B_alias_1")
            conv = next(node for node in converted.graph.node if node.name == f"conv_{index}")
            self.assertEqual(list(conv.input[1:]), ["W", "B"])

    def test_direct_initializer_needs_no_rewrite(self):
        source = conv_alias_model(weight_depth=0, bias_depth=0)
        converted, report = adapt_model(source)
        self.assertEqual(report["static_conv_weight_aliases"], {
            "rewritten_input_count": 0, "rewritten_conv_count": 0, "aliases": [],
        })
        self.assertEqual(converted.SerializeToString(), source.SerializeToString())

    def test_optional_bias_stays_absent(self):
        for empty_slot in (False, True):
            with self.subTest(empty_slot=empty_slot):
                source = conv_alias_model(with_bias=False)
                if empty_slot:
                    next(node for node in source.graph.node if node.name == "conv_0").input.append("")
                converted, report = adapt_model(source)
                conv = next(node for node in converted.graph.node if node.name == "conv_0")
                self.assertEqual(list(conv.input), ["activation_alias", "W"] + ([""] if empty_slot else []))
                self.assertEqual(report["static_conv_weight_aliases"]["rewritten_input_count"], 1)
                self.assertEqual([entry["role"] for entry in report["static_conv_weight_aliases"]["aliases"]], ["weight"])

    def test_dynamic_weight_alias_stays_dynamic(self):
        source = conv_alias_model(with_bias=False, weight_depth=2)
        source.graph.ClearField("initializer")
        source.graph.input.append(helper.make_tensor_value_info("W", TensorProto.FLOAT, [2, 2, 1, 1]))
        converted, report = adapt_model(source)
        self.assertEqual(converted.SerializeToString(), source.SerializeToString())
        self.assertEqual(report["static_conv_weight_aliases"]["rewritten_input_count"], 0)

    def test_overridable_initializer_is_not_folded(self):
        source = conv_alias_model(with_bias=False)
        source.graph.input.append(helper.make_tensor_value_info("W", TensorProto.FLOAT, [2, 2, 1, 1]))
        converted, report = adapt_model(source)
        self.assertEqual(converted.SerializeToString(), source.SerializeToString())
        self.assertEqual(report["static_conv_weight_aliases"]["rewritten_input_count"], 0)
        # The runtime override must still reach Conv through its Identity.
        feeds = {"X": np.ones((1, 2, 3, 3), dtype=np.float32),
                 "W": np.zeros((2, 2, 1, 1), dtype=np.float32)}
        np.testing.assert_array_equal(ReferenceEvaluator(converted).run(None, feeds)[0], np.zeros((1, 2, 3, 3)))

    def test_custom_identity_and_custom_conv_are_not_folded(self):
        for options in ({"identity_domain": "example.custom"}, {"conv_domain": "example.custom"}):
            with self.subTest(options=options):
                source = conv_alias_model(**options)
                converted, report = adapt_model(source)
                self.assertEqual(converted.SerializeToString(), source.SerializeToString())
                self.assertEqual(report["static_conv_weight_aliases"]["rewritten_input_count"], 0)

    def test_source_io_weights_identity_and_other_nodes_are_preserved(self):
        source = conv_alias_model(weight_depth=3, bias_depth=2, conv_count=2)
        original = source.SerializeToString()
        converted, _ = adapt_model(source)
        self.assertEqual(source.SerializeToString(), original)
        self.assertEqual(converted.graph.input, source.graph.input)
        self.assertEqual(converted.graph.output, source.graph.output)
        self.assertEqual([item.SerializeToString() for item in converted.graph.initializer],
                         [item.SerializeToString() for item in source.graph.initializer])
        self.assertEqual(len(converted.graph.node), len(source.graph.node))
        for before, after in zip(source.graph.node, converted.graph.node):
            if before.op_type != "Conv":
                self.assertEqual(after.SerializeToString(), before.SerializeToString())
            else:
                self.assertEqual(after.name, before.name)
                self.assertEqual(after.input[0], before.input[0])
                self.assertEqual(after.output, before.output)
                self.assertEqual(after.attribute, before.attribute)

    def test_reference_evaluator_random_conv_results_are_exact(self):
        source = conv_alias_model(weight_depth=3, bias_depth=2, conv_count=2)
        converted, _ = adapt_model(source)
        original_eval, converted_eval = ReferenceEvaluator(source), ReferenceEvaluator(converted)
        rng = np.random.default_rng(81)
        for sample in range(3):
            with self.subTest(sample=sample):
                feeds = {"X": rng.normal(size=(1, 2, 3, 3)).astype(np.float32)}
                for original, actual in zip(original_eval.run(None, feeds), converted_eval.run(None, feeds)):
                    np.testing.assert_array_equal(actual, original)


if __name__ == "__main__":
    unittest.main()
