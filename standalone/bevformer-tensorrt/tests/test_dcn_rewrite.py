"""CPU tests for explicit DCN substitution; no CUDA accuracy claims."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from bevformer_trt.dcn import (GAC_PLUGIN, NVIDIA_PLUGIN, PLUGIN_DOMAIN, prepare_parser_input,
                              prove_dcn_only, read_config, rewrite_for_parser, rewrite_model,
                              validate_config)


def fixture():
    nodes = [helper.make_node(NVIDIA_PLUGIN, ["image", "offset", "mask", "weight"], [f"dcn{n}"],
                             name=f"camera{n}/dcn", domain=PLUGIN_DOMAIN, plugin_version="2",
                             plugin_namespace="", stride=[1, 1], padding=[1, 1], dilation=[1, 1],
                             group=1, deformable_group=1) for n in range(2)]
    nodes.append(helper.make_node("Add", ["dcn0", "dcn1"], ["output"], name="head"))
    nodes[0].doc_string = "retain node metadata"
    graph = helper.make_graph(nodes, "dcn-experiment", [
        helper.make_tensor_value_info("image", TensorProto.FLOAT, [1, 4, 5, 5]),
        helper.make_tensor_value_info("offset", TensorProto.FLOAT, [1, 18, 5, 5]),
        helper.make_tensor_value_info("mask", TensorProto.FLOAT, [1, 9, 5, 5])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 8, 5, 5])],
        initializer=[numpy_helper.from_array(np.arange(8*4*9, dtype=np.float32).reshape(8, 4, 3, 3), "weight")])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17),
                                                  helper.make_opsetid(PLUGIN_DOMAIN, 1)])
    model.metadata_props.add(key="note", value="unchanged model metadata")
    return model


def config():
    return {"node_names": ["camera0/dcn"], "block_threads": 256, "specialize": 3}


def attr(node, name):
    return next(a for a in node.attribute if a.name == name)


class DcnRewriteTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)

    def test_explicit_subset_preserves_full_graph_and_source(self):
        source = fixture()
        original = source.SerializeToString()
        custom, audit = rewrite_model(source, config())
        self.assertEqual(source.SerializeToString(), original)
        self.assertEqual(custom.graph.node[0].op_type, GAC_PLUGIN)
        self.assertEqual(custom.graph.node[1:], source.graph.node[1:])
        self.assertEqual(audit["replaced_node_count"], 1)
        self.assertEqual(audit["records"][0]["known_weight_shape"], [8, 4, 3, 3])
        self.assertEqual(audit["validation"]["numerical_accuracy"], "not_checked")
        self.assertEqual(attr(custom.graph.node[0], "specialize").i, 3)
        self.assertEqual(attr(custom.graph.node[0], "plugin_version").s, b"1")
        restored = deepcopy(custom)
        restored.graph.node[0].CopyFrom(source.graph.node[0])
        self.assertEqual(restored.SerializeToString(), original)
        onnx.checker.check_model(custom)

    def test_all_supported_fields_roundtrip(self):
        for block in (128, 256, 512, 1024):
            for specialize in (1, 3, 4):
                with self.subTest(block=block, specialize=specialize):
                    selected = {**config(), "block_threads": block, "specialize": specialize}
                    output, audit = rewrite_for_parser(fixture().SerializeToString(), selected)
                    proof = prove_dcn_only(fixture(), onnx.load_model_from_string(output))
                    self.assertEqual(proof, audit["records"])

    def test_default_is_no_substitution(self):
        original = fixture().SerializeToString()
        output, audit = rewrite_for_parser(original, None)
        self.assertIs(output, original)
        self.assertFalse(audit["enabled"])
        source = self.directory / "original.onnx"
        source.write_bytes(original)
        output, audit = prepare_parser_input(source)
        self.assertEqual(output, original)
        self.assertEqual(audit["source_onnx_sha256"], audit["transformed_onnx_sha256"])
        self.assertEqual(audit["records"], [])

    def test_exact_selection_rejects_missing_duplicate_wrong_and_ambiguous(self):
        for names in (["camera"], ["camera0/dcn", "camera0/dcn"], ["head"], [], "camera0/dcn", [3], [""]):
            with self.subTest(names=names), self.assertRaises(ValueError):
                rewrite_model(fixture(), {**config(), "node_names": names})
        model = fixture()
        model.graph.node[1].name = "camera0/dcn"
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            rewrite_model(model, config())

    def test_rejects_coerced_unknown_or_missing_options(self):
        for bad in ({**config(), "block_threads": True}, {**config(), "specialize": 3.0},
                    {**config(), "specialize": 0}, {**config(), "block_threads": 768},
                    {**config(), "extra": 1}, {"node_names": ["camera0/dcn"]}, None):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                validate_config(bad)

    def test_config_json_rejects_duplicate_keys(self):
        path = self.directory / "config.json"
        path.write_text('{"node_names":["camera0/dcn"],"block_threads":256,"specialize":3,"specialize":4}')
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            read_config(path)
        path.write_text(json.dumps(config()), encoding="utf-8-sig")
        self.assertEqual(read_config(path), config())

    def test_rejects_unsupported_plugin_contract(self):
        mutations = [lambda n: setattr(n, "domain", "other"),
                     lambda n: setattr(attr(n, "plugin_version"), "s", b"1"),
                     lambda n: setattr(attr(n, "plugin_namespace"), "s", b"other"),
                     lambda n: n.input.append("bias"),
                     lambda n: n.input.__setitem__(0, ""),
                     lambda n: n.output.append("extra"),
                     lambda n: setattr(attr(n, "group"), "i", 2),
                     lambda n: setattr(attr(n, "deformable_group"), "i", 2),
                     lambda n: attr(n, "stride").ints.__setitem__(0, 0),
                     lambda n: attr(n, "padding").ints.__setitem__(0, -1),
                     lambda n: attr(n, "dilation").ints.append(1),
                     lambda n: n.attribute.append(helper.make_attribute("unknown", 1)),
                     lambda n: n.attribute.append(deepcopy(attr(n, "group")))]
        for mutate in mutations:
            model = fixture()
            mutate(model.graph.node[0])
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                rewrite_model(model, config())

    def test_known_non3x3_rejected_and_unknown_shape_deferred(self):
        model = fixture()
        model.graph.initializer[0].dims[-1] = 1
        with self.assertRaisesRegex(ValueError, "3x3"):
            rewrite_model(model, config())
        model = fixture()
        model.graph.ClearField("initializer")
        _, audit = rewrite_model(model, config())
        self.assertIsNone(audit["records"][0]["known_weight_shape"])

    def test_proof_rejects_semantic_edges_weights_shapes_and_metadata_changes(self):
        source = fixture()
        custom, _ = rewrite_model(source, config())
        mutations = [lambda m: attr(m.graph.node[0], "stride").ints.__setitem__(0, 2),
                     lambda m: setattr(attr(m.graph.node[0], "stride"), "doc_string", "changed"),
                     lambda m: m.graph.node[0].input.__setitem__(0, "changed"),
                     lambda m: m.graph.node[0].output.__setitem__(0, "changed"),
                     lambda m: setattr(m.graph.node[0], "doc_string", "changed"),
                     lambda m: setattr(m.graph.initializer[0], "raw_data", b"changed"),
                     lambda m: setattr(m.graph.node[-1], "op_type", "Sub"),
                     lambda m: setattr(m, "doc_string", "changed"),
                     lambda m: setattr(m.graph.output[0].type.tensor_type.shape.dim[0], "dim_value", 2)]
        for mutate in mutations:
            damaged = deepcopy(custom)
            mutate(damaged)
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                prove_dcn_only(source, damaged)

    def test_proof_rejects_extra_nodes_and_invalid_custom_fields(self):
        source = fixture()
        custom, _ = rewrite_model(source, config())
        damaged = deepcopy(custom)
        damaged.graph.node.append(helper.make_node("Identity", ["image"], ["extra"]))
        with self.assertRaisesRegex(ValueError, "node count"):
            prove_dcn_only(source, damaged)
        attr(custom.graph.node[0], "block_threads").i = 768
        with self.assertRaisesRegex(ValueError, "block_threads"):
            prove_dcn_only(source, custom)
        with self.assertRaisesRegex(ValueError, "No explicit"):
            prove_dcn_only(source, source)

    def test_external_weights_and_local_functions_rejected(self):
        source = fixture()
        source.graph.initializer[0].data_location = TensorProto.EXTERNAL
        with self.assertRaisesRegex(ValueError, "embedded"):
            rewrite_model(source, config())
        source = fixture()
        source.functions.add(name="local", domain="test")
        with self.assertRaisesRegex(ValueError, "inference graph"):
            rewrite_model(source, config())

    def test_file_api_hashes_without_mutating_source_or_config(self):
        source, settings = self.directory / "model.onnx", self.directory / "config.json"
        original = fixture().SerializeToString()
        original_config = json.dumps(config()).encode()
        source.write_bytes(original)
        settings.write_bytes(original_config)
        payload, audit = prepare_parser_input(source, settings)
        self.assertEqual(source.read_bytes(), original)
        self.assertEqual(settings.read_bytes(), original_config)
        self.assertEqual(audit["source_onnx_sha256"], hashlib.sha256(original).hexdigest())
        self.assertEqual(audit["transformed_onnx_sha256"], hashlib.sha256(payload).hexdigest())
        self.assertEqual(audit["config_sha256"], hashlib.sha256(original_config).hexdigest())
        self.assertFalse(audit["frozen_calibration"]["recalibrated"])
        self.assertEqual(len(list(self.directory.iterdir())), 2)


if __name__ == "__main__":
    unittest.main()
