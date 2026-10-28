#!/usr/bin/env python3
"""CPU-only tests for exact backbone extraction, not TensorRT calibration.

The six-camera toy has ordinary 1x1 convolutions so ONNX ReferenceEvaluator can
check its actual outputs. It is deliberately not a claim of Base/DCNv2 accuracy.
Run: python -m unittest discover -s tests -p test_backbone_calibration.py
"""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from onnx.reference import ReferenceEvaluator

from bevformer_trt.backbone_calibration import extract_model, sha256_file, write_subgraph
from bevformer_trt.precision_policy import policy_fingerprint


TOY_INPUT_SHAPE = (1, 6, 1, 2, 2)
TOY_OUTPUT_SHAPE = (1, 1, 2, 2)


def make_model():
    """Six disjoint backbones followed by six expendable head convolutions."""
    nodes, weights, outputs, terminal_names, int8_names = [], [], [], [], []
    for camera in range(6):
        prefix = f"camera_{camera}"
        stem = f"/img_backbone_{camera}/conv1/Conv"
        body = f"/img_backbone_{camera}/layer4/layer4.2/conv3/Conv"
        terminal = f"/img_backbone_{camera}/layer4/layer4.2/relu/Relu"
        names = {key: f"{prefix}_{key}" for key in (
            "index", "image", "stem", "body", "features", "head",
            "stem_weight", "stem_bias", "body_weight", "head_weight",
        )}
        arrays = {
            "index": np.array(camera, dtype=np.int64),
            "stem_weight": np.full((1, 1, 1, 1), camera + 1, dtype=np.float32),
            "stem_bias": np.array([-0.25 * camera], dtype=np.float32),
            "body_weight": np.full((1, 1, 1, 1), 0.5, dtype=np.float32),
            "head_weight": np.full((2, 1, 1, 1), 3.0, dtype=np.float32),
        }
        weights.extend(numpy_helper.from_array(array, names[key]) for key, array in arrays.items())
        nodes.extend([
            helper.make_node("Gather", ["images", names["index"]], [names["image"]],
                             name=f"/camera_slice_{camera}", axis=1),
            helper.make_node("Conv", [names["image"], names["stem_weight"], names["stem_bias"]],
                             [names["stem"]], name=stem),
            helper.make_node("Conv", [names["stem"], names["body_weight"]],
                             [names["body"]], name=body),
            helper.make_node("Relu", [names["body"]], [names["features"]], name=terminal),
            helper.make_node("Conv", [names["features"], names["head_weight"]],
                             [names["head"]], name=f"/img_neck_{camera}/Conv"),
        ])
        outputs.append(helper.make_tensor_value_info(names["head"], TensorProto.FLOAT, (1, 2, 2, 2)))
        terminal_names.append(names["features"])
        int8_names.append(body)
    weights.append(numpy_helper.from_array(np.array([123.0], dtype=np.float32), "unused_weight"))
    graph = helper.make_graph(
        nodes, "six_camera_backbone_toy",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, TOY_INPUT_SHAPE)],
        outputs, initializer=weights,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    onnx.checker.check_model(model)
    return model, terminal_names, int8_names


class BackboneExtractionTests(unittest.TestCase):
    def setUp(self):
        self.model, self.terminal_names, self.int8_names = make_model()

    def extract(self, **changes):
        options = dict(expected_input_shape=TOY_INPUT_SHAPE, expected_output_count=6,
                       expected_int8_count=6, expected_int8_names=self.int8_names)
        options.update(changes)
        return extract_model(self.model, **options)

    def test_only_six_terminal_outputs_and_exact_int8_coverage(self):
        submodel, audit = self.extract()
        self.assertEqual([v.name for v in submodel.graph.output], self.terminal_names)
        self.assertEqual(audit["int8_convolution_names"], self.int8_names)
        self.assertEqual(audit["coverage"]["retained_int8_convolutions"], 6)
        self.assertEqual(audit["coverage"]["missing"], [])
        self.assertEqual(audit["coverage"]["extra"], [])
        self.assertEqual(len(audit["coverage"]["per_output"]), 6)
        for index, branch in enumerate(audit["coverage"]["per_output"]):
            self.assertEqual(branch["output_tensor"], self.terminal_names[index])
            self.assertEqual(branch["int8_convolution_names"], [self.int8_names[index]])
        conv_outputs = {name for n in submodel.graph.node if n.op_type == "Conv" for name in n.output}
        self.assertTrue(conv_outputs.isdisjoint(self.terminal_names))
        self.assertEqual(sum(n.op_type == "Conv" for n in submodel.graph.node), 12)

    def test_output_shapes_and_identity_tensor_mapping(self):
        submodel, audit = self.extract()
        self.assertEqual(audit["inputs"], [{"name": "images", "dtype": "float32", "shape": list(TOY_INPUT_SHAPE)}])
        for output, description in zip(submodel.graph.output, audit["outputs"]):
            self.assertEqual([d.dim_value for d in output.type.tensor_type.shape.dim], list(TOY_OUTPUT_SHAPE))
            self.assertEqual(description["shape"], list(TOY_OUTPUT_SHAPE))
            self.assertEqual(description["dtype"], "float32")
        mapping = {item["source_name"]: item for item in audit["tensor_mapping"]}
        for name in ["images", *self.terminal_names]:
            self.assertIn(name, mapping)
            self.assertEqual(mapping[name]["subgraph_name"], name)
        for item in mapping.values():
            self.assertEqual(item["source_name"], item["subgraph_name"])

    def test_required_ranges_cover_only_int8_activation_edges(self):
        submodel, audit = self.extract()
        expected = [
            {"layer": node.name, "edge": edge, "tensor": tensor}
            for node in submodel.graph.node if node.name in self.int8_names
            for edge, tensor in (("input0", node.input[0]), ("output0", node.output[0]))
        ]
        self.assertEqual(audit["int8_activation_edges"], expected)
        self.assertEqual(len(expected), 12)
        mapped = {item["source_name"] for item in audit["tensor_mapping"]}
        self.assertTrue(all(item["tensor"] in mapped for item in expected))

    def test_source_and_retained_protobufs_are_unchanged(self):
        original = self.model.SerializeToString()
        submodel, audit = self.extract()
        self.assertEqual(self.model.SerializeToString(), original)
        source_nodes = {node.name: node.SerializeToString() for node in self.model.graph.node}
        source_weights = {weight.name: weight.SerializeToString() for weight in self.model.graph.initializer}
        for node in submodel.graph.node:
            self.assertEqual(node.SerializeToString(), source_nodes[node.name])
        for weight in submodel.graph.initializer:
            self.assertEqual(weight.SerializeToString(), source_weights[weight.name])
        self.assertTrue(all(audit["preservation"].values()))
        self.assertEqual(submodel.graph.input, self.model.graph.input)

    def test_heads_and_unused_weights_are_removed(self):
        submodel, audit = self.extract()
        self.assertFalse(any("img_neck" in node.name for node in submodel.graph.node))
        self.assertFalse(any(weight.name.endswith("head_weight") or weight.name == "unused_weight"
                             for weight in submodel.graph.initializer))
        self.assertFalse(any("head" in item["source_name"] for item in audit["tensor_mapping"]))
        self.assertEqual(len(submodel.graph.node), 24)
        self.assertEqual(len(submodel.graph.initializer), 24)

    def test_reference_evaluator_matches_original_terminal_values(self):
        submodel, _ = self.extract()
        oracle = deepcopy(self.model)
        oracle.graph.ClearField("output")
        oracle.graph.output.extend(helper.make_tensor_value_info(name, TensorProto.FLOAT, TOY_OUTPUT_SHAPE)
                                   for name in self.terminal_names)
        # Both signs exercise ReLU; differing camera weights catch wrong routing.
        images = np.linspace(-2.0, 3.0, np.prod(TOY_INPUT_SHAPE), dtype=np.float32).reshape(TOY_INPUT_SHAPE)
        expected = ReferenceEvaluator(oracle).run(None, {"images": images})
        actual = ReferenceEvaluator(submodel).run(None, {"images": images})
        self.assertEqual(len(actual), 6)
        for original, extracted in zip(expected, actual):
            np.testing.assert_array_equal(extracted, original)

    def test_incorrect_expected_int8_names_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "INT8 Conv names disagree"):
            self.extract(expected_int8_names=[*self.int8_names[:-1], "/wrong/Conv"])

    def test_missing_terminal_branch_is_rejected(self):
        next(node for node in self.model.graph.node if node.output[0] == self.terminal_names[-1]).op_type = "Identity"
        with self.assertRaisesRegex(ValueError, "Expected 6 final layer4 outputs, found 5"):
            self.extract()

    def test_extra_graph_input_is_rejected(self):
        self.model.graph.input.append(helper.make_tensor_value_info("unexpected", TensorProto.FLOAT, [1]))
        with self.assertRaisesRegex(ValueError, "exactly one graph input"):
            self.extract()

    def test_duplicate_tensor_producer_is_rejected(self):
        self.model.graph.node.append(helper.make_node("Identity", ["images"], ["camera_0_body"], name="duplicate"))
        with self.assertRaisesRegex(ValueError, "Duplicate tensor producer: camera_0_body"):
            self.extract()

    def test_unadapted_dcn_is_rejected(self):
        stem = next(node for node in self.model.graph.node if node.name == "/img_backbone_0/conv1/Conv")
        stem.domain, stem.op_type = "mmcv", "MMCVModulatedDeformConv2d"
        self.model.opset_import.append(helper.make_opsetid("mmcv", 1))
        with self.assertRaisesRegex(ValueError, "requires an adapted DCNv2 plugin"):
            self.extract()

    def test_adapted_dcn_shape_inference_preserves_saved_plugin(self):
        stem = next(node for node in self.model.graph.node if node.name == "/img_backbone_0/conv1/Conv")
        stem.CopyFrom(helper.make_node(
            "ModulatedDeformConv2d",
            ["camera_0_image", "camera_0_offsets", "camera_0_mask", "camera_0_stem_weight"],
            ["camera_0_stem"], name=stem.name, domain="trt.plugins",
            plugin_version="2", plugin_namespace="", stride=[1, 1], padding=[0, 0],
            dilation=[1, 1], group=1, deformable_group=1,
        ))
        self.model.graph.initializer.extend([
            numpy_helper.from_array(np.zeros((1, 2, 2, 2), dtype=np.float32), "camera_0_offsets"),
            numpy_helper.from_array(np.ones((1, 1, 2, 2), dtype=np.float32), "camera_0_mask"),
        ])
        self.model.opset_import.append(helper.make_opsetid("trt.plugins", 1))
        original_model = self.model.SerializeToString()
        original_plugin = stem.SerializeToString()

        submodel, audit = self.extract()
        # Inspect the serialized artifact, not the disposable shape-only model.
        saved = onnx.load_model_from_string(submodel.SerializeToString())
        saved_plugin = next(node for node in saved.graph.node if node.name == stem.name)
        self.assertEqual((saved_plugin.domain, saved_plugin.op_type),
                         ("trt.plugins", "ModulatedDeformConv2d"))
        self.assertEqual(saved_plugin.SerializeToString(), original_plugin)
        self.assertEqual(self.model.SerializeToString(), original_model)
        for output in audit["outputs"]:
            self.assertEqual(output["shape"], list(TOY_OUTPUT_SHAPE))
            self.assertEqual(output["dtype"], "float32")
        stem_info = next(item for item in audit["tensor_mapping"] if item["source_name"] == "camera_0_stem")
        self.assertEqual(stem_info["shape"], list(TOY_OUTPUT_SHAPE))
        self.assertEqual(stem_info["dtype"], "float32")

    def test_duplicate_int8_names_are_rejected(self):
        next(node for node in self.model.graph.node if node.name == self.int8_names[1]).name = self.int8_names[0]
        with self.assertRaisesRegex(ValueError, "INT8 Conv node names must be unique"):
            self.extract()

    def test_external_weights_are_rejected(self):
        self.model.graph.initializer[0].data_location = TensorProto.EXTERNAL
        with self.assertRaisesRegex(ValueError, "requires embedded weights"):
            self.extract()

    def test_int8_convolution_outside_extracted_ancestors_is_rejected(self):
        orphan_name = "/img_backbone_0/layer3/orphan/Conv"
        self.model.graph.node.append(helper.make_node("Conv", ["camera_0_image", "camera_0_body_weight"],
                                                     ["orphan"], name=orphan_name))
        with self.assertRaisesRegex(ValueError, "Extraction loses INT8 Conv nodes"):
            self.extract(expected_int8_count=7, expected_int8_names=[*self.int8_names, orphan_name])


class BackboneWriterTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        self.source, self.output = root / "source.onnx", root / "subgraph.onnx"
        self.manifest, self.plan = root / "manifest.json", root / "plan.json"
        model, _, int8_names = make_model()
        onnx.save(model, self.source)
        self.plan.write_text(json.dumps({
            "onnx_file_sha256": sha256_file(self.source),
            "policy_fingerprint": "test-only-full-policy-fingerprint",
            "precision_policy": {"mixed_layer_policy": {
                "fingerprint_sha256": policy_fingerprint(),
                "layers": [{"name": name, "compute_precision": "int8"} for name in int8_names],
            }},
        }), encoding="utf-8")

    def write(self, **changes):
        options = dict(source=self.source, output=self.output, manifest_path=self.manifest, precision_plan=self.plan)
        options.update(changes)
        return write_subgraph(**options)

    def test_cli_path_does_not_accept_toy_contract(self):
        # The public writer must keep Base's real shape/count requirements.
        with self.assertRaisesRegex(ValueError, "Unexpected images input contract"):
            self.write()
        self.assertFalse(self.output.exists())
        self.assertFalse(self.manifest.exists())

    def test_cli_path_requires_all_462_int8_convolutions(self):
        model = onnx.load(self.source)
        model.graph.input[0].CopyFrom(helper.make_tensor_value_info(
            "images", TensorProto.FLOAT, (1, 6, 3, 928, 1600)))
        onnx.save(model, self.source)
        plan = json.loads(self.plan.read_text(encoding="utf-8"))
        plan["onnx_file_sha256"] = sha256_file(self.source)
        self.plan.write_text(json.dumps(plan), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Expected 462 source INT8 Conv nodes, found 6"):
            self.write()
        self.assertFalse(self.output.exists())
        self.assertFalse(self.manifest.exists())

    def test_colliding_paths_are_rejected_without_writes(self):
        original_source, original_plan = self.source.read_bytes(), self.plan.read_bytes()
        for changes in ({"output": self.source}, {"manifest_path": self.source},
                        {"precision_plan": self.source}, {"manifest_path": self.output}):
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, "paths must differ"):
                self.write(**changes)
        self.assertEqual(self.source.read_bytes(), original_source)
        self.assertEqual(self.plan.read_bytes(), original_plan)

    def test_existing_output_or_manifest_is_never_overwritten(self):
        for destination in (self.output, self.manifest):
            with self.subTest(destination=destination):
                destination.write_bytes(b"preserve this file")
                with self.assertRaises(FileExistsError):
                    self.write()
                self.assertEqual(destination.read_bytes(), b"preserve this file")
                destination.unlink()

    def test_precision_plan_for_different_onnx_is_rejected(self):
        plan = json.loads(self.plan.read_text(encoding="utf-8"))
        plan["onnx_file_sha256"] = "0" * 64
        self.plan.write_text(json.dumps(plan), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "not bound to the source full ONNX"):
            self.write()
        self.assertFalse(self.output.exists())
        self.assertFalse(self.manifest.exists())


if __name__ == "__main__":
    unittest.main()
