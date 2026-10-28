"""CPU structural/provenance tests; these do not validate CUDA plugin accuracy."""

from copy import deepcopy
import ctypes
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from bevformer_trt import engine
from bevformer_trt.calibration_ranges import FORMAT_KIND, PRODUCTION_KIND, TRT_VERSION, apply_calibrated_ranges
from bevformer_trt.msda import (DEFAULT_FIELDS, GAC_PLUGIN, NVIDIA_PLUGIN, PLUGIN_DOMAIN,
                                prove_msda_only, rebind_ranges, rewrite_file, rewrite_model)
from bevformer_trt.precision_policy import policy_fingerprint


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(value).hexdigest()


def model_fixture(count=2, backbone=False):
    initializers = [numpy_helper.from_array(array, name) for name, array in {
        "value": np.zeros((1, 4, 1, 4), np.float32),
        "spatial": np.array([[2, 2]], np.int32), "starts": np.array([0], np.int32),
        "locations": np.zeros((1, 1, 1, 1, 1, 2), np.float32),
        "attention": np.ones((1, 1, 1, 1, 1), np.float32),
        "kernel": np.ones((3, 3, 1, 1), np.float32),
    }.items()]
    nodes = []
    if backbone:
        nodes.append(helper.make_node("ReduceMean", ["images"], ["camera_mean"], name="camera_mean", axes=[1], keepdims=0))
        previous = "camera_mean"
        for index in range(462):
            name = f"/img_backbone/layer1/unit{index}/Conv"
            nodes.append(helper.make_node("Conv", [previous, "kernel"], [f"conv_{index}"], name=name))
            previous = f"conv_{index}"
    for index in range(count):
        nodes.append(helper.make_node(NVIDIA_PLUGIN, ["value", "spatial", "starts", "locations", "attention"],
                     [f"msda_{index}"], domain=PLUGIN_DOMAIN, name=f"attention_{index}",
                     plugin_version="2", plugin_namespace=""))
        nodes[-1].doc_string = "Preserve this original node metadata"
    nodes.append(helper.make_node("Identity", ["msda_0"], ["result"], name="downstream"))
    graph = helper.make_graph(nodes, "audited_msda", [helper.make_tensor_value_info("images", TensorProto.FLOAT,
        [1, 6, 3, 928, 1600])], [helper.make_tensor_value_info("result", TensorProto.FLOAT, [1, 1, 1, 4])],
        initializer=initializers, value_info=[helper.make_tensor_value_info("msda_0", TensorProto.FLOAT, [1, 1, 1, 4])])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid(PLUGIN_DOMAIN, 1)])
    model.metadata_props.add(key="producer-note", value="preserve exact metadata")
    return model


class RewriteTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_exact_structure_weights_shape_and_subset_preservation(self):
        source = model_fixture()
        original = source.SerializeToString()
        result, report = rewrite_model(source, node_names=["attention_1"], expected_msda_count=2,
                                      node_configs={"attention_1": {"channels_per_thread": 4, "prefetch": 1}})
        self.assertEqual(source.SerializeToString(), original)
        self.assertEqual(report["replaced_node_count"], 1)
        self.assertEqual(report["nodes"][0]["fields"], {**DEFAULT_FIELDS, "channels_per_thread": 4, "prefetch": 1})
        self.assertEqual(result.graph.node[0], source.graph.node[0])
        self.assertEqual(result.graph.node[1].op_type, GAC_PLUGIN)
        self.assertEqual(result.graph.node[-1], source.graph.node[-1])
        self.assertEqual(result.graph.initializer, source.graph.initializer)
        self.assertEqual(result.graph.value_info, source.graph.value_info)
        self.assertEqual(result.graph.input, source.graph.input)
        self.assertEqual(result.graph.output, source.graph.output)
        restored = deepcopy(result)
        restored.graph.node[1].CopyFrom(source.graph.node[1])
        self.assertEqual(restored.SerializeToString(), original)
        onnx.checker.check_model(result)

    def test_default_all_eighteen_and_exact_selection(self):
        result, report = rewrite_model(model_fixture(18))
        self.assertEqual(report["replaced_node_count"], 18)
        self.assertTrue(all(node.op_type == GAC_PLUGIN for node in result.graph.node[:-1]))
        for names in (["attention"], ["attention_0", "attention_0"], ["downstream"]):
            with self.subTest(names=names), self.assertRaises(ValueError):
                rewrite_model(model_fixture(), node_names=names, expected_msda_count=2)
        with self.assertRaisesRegex(ValueError, "Expected 18"):
            rewrite_model(model_fixture())

    def test_guarded_index_and_value_prefetch_roundtrip(self):
        source = model_fixture()
        fields = {"channels_per_thread": 4, "block_threads": 128, "specialize": 3, "prefetch": 2}
        rewritten, report = rewrite_model(source, fields=fields, expected_msda_count=2)
        restored = onnx.load_model_from_string(rewritten.SerializeToString())
        proof = prove_msda_only(source, restored)
        self.assertEqual(len(proof), 2)
        self.assertTrue(all(node["fields"] == fields for node in proof))
        self.assertEqual(report["nodes"], proof)

    def test_invalid_version_abi_namespace_and_fields_rejected(self):
        for mutation in (lambda node: setattr(node, "domain", "other"),
                         lambda node: setattr(node.attribute[0], "s", b"unexpected"),
                         lambda node: node.input.pop(),
                         lambda node: node.attribute.add(name="unknown", type=2, i=1)):
            model = model_fixture()
            mutation(model.graph.node[0])
            with self.assertRaises(ValueError):
                rewrite_model(model, node_names=["attention_0"], expected_msda_count=None)
        for fields in ({"channels_per_thread": 3}, {"prefetch": True}, {"block_threads": 64}, {"extra": 1},
                       {"specialize": 4}, {"prefetch": 3}, {"specialize": 3.0}):
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                rewrite_model(model_fixture(), fields=fields, expected_msda_count=2)

    def test_proof_rejects_unrelated_or_edge_changes(self):
        source = model_fixture()
        rewritten, _ = rewrite_model(source, expected_msda_count=2)
        mutations = [lambda model: setattr(model, "doc_string", "changed"),
                     lambda model: setattr(model.graph.initializer[0], "raw_data", b"changed"),
                     lambda model: setattr(model.graph.node[-1], "op_type", "Relu"),
                     lambda model: model.graph.node[0].input.__setitem__(0, "wrong"),
                     lambda model: setattr(model.graph.output[0].type.tensor_type.shape.dim[0], "dim_value", 2)]
        for mutate in mutations:
            damaged = deepcopy(rewritten)
            mutate(damaged)
            with self.assertRaises(ValueError):
                prove_msda_only(source, damaged)

    def test_file_hashes_and_no_overwrite(self):
        source, output, report = (self.root / name for name in ("source.onnx", "custom.onnx", "rewrite.json"))
        onnx.save(model_fixture(), source)
        audit = rewrite_file(source, output, report, expected_msda_count=2)
        self.assertEqual(audit["source_onnx_sha256"], digest(source.read_bytes()))
        self.assertEqual(audit["output_onnx_sha256"], digest(output.read_bytes()))
        saved = output.read_bytes()
        with self.assertRaises(FileExistsError):
            rewrite_file(source, output, report, expected_msda_count=2)
        self.assertEqual(output.read_bytes(), saved)
        with self.assertRaises(ValueError):
            rewrite_file(source, source, self.root / "new.json", expected_msda_count=2)
        self.assertFalse((self.root / "new.json").exists())

    def test_external_weights_rejected(self):
        model = model_fixture()
        model.graph.initializer[0].data_location = TensorProto.EXTERNAL
        with self.assertRaisesRegex(ValueError, "embedded weights"):
            rewrite_model(model, expected_msda_count=2)

    def test_unmodified_head_control_flow_is_preserved(self):
        source = model_fixture()
        branch = helper.make_graph([helper.make_node("Identity", ["result"], ["branch_result"])], "branch", [],
            [helper.make_tensor_value_info("branch_result", TensorProto.FLOAT, [1, 1, 1, 4])])
        source.graph.node.append(helper.make_node("If", ["condition"], ["conditional_result"],
            name="head_conditional", then_branch=branch, else_branch=branch))
        rewritten, _ = rewrite_model(source, expected_msda_count=2)
        self.assertEqual(source.graph.node[-1].SerializeToString(), rewritten.graph.node[-1].SerializeToString())
        rewritten.graph.node[-1].attribute[0].g.name = "changed"
        with self.assertRaises(ValueError):
            prove_msda_only(source, rewritten)


class RangeRebindingTests(unittest.TestCase):
    """Production-sized contract fixtures; no actual calibration is claimed."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source, self.destination, self.ranges, self.output = (
            self.root / name for name in ("source.onnx", "destination.onnx", "ranges.json", "rebound.json"))
        source = model_fixture(backbone=True)
        onnx.save(source, self.source)
        rewritten, _ = rewrite_model(source, expected_msda_count=2)
        onnx.save(rewritten, self.destination)
        names = [node.name for node in source.graph.node if node.op_type == "Conv"]
        mapping = [{"source_name": name, "subgraph_name": name, "dtype": "float32"}
                   for name in ["images", "camera_mean", *[f"conv_{index}" for index in range(462)]]]
        manifest = {"schema_version": 1, "kind": PRODUCTION_KIND,
            "source_full_onnx_sha256": digest(self.source.read_bytes()), "subgraph_onnx_sha256": "a" * 64,
            "policy_rule_fingerprint_sha256": policy_fingerprint(), "int8_convolution_names": names,
            "coverage": {"expected_int8_convolutions": 462, "retained_int8_convolutions": 462, "missing": [], "extra": []},
            "preservation": {key: True for key in ("node_protobufs_identical", "initializer_protobufs_identical",
                            "input_value_info_identical", "tensor_names_identical")},
            "source_full_policy_fingerprint": "b" * 64, "precision_plan_sha256": "c" * 64,
            "inputs": [{"name": "images", "dtype": "float32", "shape": [1, 6, 3, 928, 1600]}],
            "tensor_mapping": mapping,
            "int8_activation_edges": [{"layer": node.name, "edge": edge, "tensor": name}
                for node in source.graph.node if node.op_type == "Conv"
                for edge, name in (("input0", node.input[0]), ("output0", node.output[0]))]}
        binding = {"model_sha256": "a" * 64, "tensorrt": TRT_VERSION, "algorithm": "IInt8EntropyCalibrator2",
                   "calibration": {"provenance": {"source_kind": "real_preprocessed"}}}
        self.artifact = {"schema_version": 1, "kind": FORMAT_KIND, "tensorrt": TRT_VERSION,
            "source_full_onnx_sha256": manifest["source_full_onnx_sha256"], "subgraph_onnx_sha256": "a" * 64,
            "policy_rule_fingerprint_sha256": policy_fingerprint(), "synthetic_interface_test": False,
            "source_manifest": {"sha256": digest(canonical(manifest)), "document": manifest},
            "source_cache": {"binding": binding, "binding_fingerprint": digest(canonical(binding)), "sha256": "e" * 64},
            "ranges": {row["source_name"]: {"scale": 0.01, "max_abs": 1.27} for row in mapping},
            "calibration_sample_count": 1}
        self.save_ranges()

    def save_ranges(self):
        self.artifact.pop("artifact_sha256", None)
        self.artifact["artifact_sha256"] = digest(canonical(self.artifact))
        self.ranges.write_text(json.dumps(self.artifact), encoding="utf-8")

    def rebind(self):
        return rebind_ranges(source_onnx=self.source, destination_onnx=self.destination,
                             ranges=self.ranges, output=self.output)

    def test_explicit_rebinding_preserves_original_cache_and_range_evidence(self):
        result = self.rebind()
        self.assertEqual(result["source_full_onnx_sha256"], digest(self.destination.read_bytes()))
        self.assertEqual(result["source_cache"], self.artifact["source_cache"])
        self.assertEqual(result["ranges"], self.artifact["ranges"])
        self.assertEqual(result["msda_rebinding"]["original_artifact"], self.artifact)
        self.assertEqual(result["msda_rebinding"]["source_onnx_sha256"], digest(self.source.read_bytes()))
        self.assertEqual(result["source_manifest"]["document"]["subgraph_onnx_sha256"], "a" * 64)
        unsigned = {key: value for key, value in result.items() if key != "artifact_sha256"}
        self.assertEqual(result["artifact_sha256"], digest(canonical(unsigned)))
        with self.assertRaises(FileExistsError):
            self.rebind()

    def test_original_ranges_still_rejected_for_new_model(self):
        trt = SimpleNamespace(__version__=TRT_VERSION)
        with self.assertRaisesRegex(ValueError, "does not match destination"):
            apply_calibrated_ranges(None, trt, path=self.ranges,
                model_sha256=digest(self.destination.read_bytes()), precision_policy={})

    def test_derived_ranges_pass_unchanged_strict_application_contract(self):
        self.rebind()
        manifest = self.artifact["source_manifest"]["document"]
        tensors = {name: SimpleNamespace(name=name, dtype="float32", is_shape_tensor=False, dynamic_range=None)
                   for name in self.artifact["ranges"]}
        layers = []
        for index, name in enumerate(manifest["int8_convolution_names"]):
            input_name = "camera_mean" if index == 0 else f"conv_{index - 1}"
            inputs = [tensors[input_name], None]
            outputs = [tensors[f"conv_{index}"]]
            layers.append(SimpleNamespace(name=name, type="conv", precision="int8", num_inputs=2, num_outputs=1,
                get_input=lambda slot, values=inputs: values[slot], get_output=lambda slot, values=outputs: values[slot]))
        network = SimpleNamespace(num_layers=len(layers), get_layer=lambda index: layers[index], num_inputs=0)
        trt = SimpleNamespace(__version__=TRT_VERSION, float16="float16", float32="float32", int8="int8",
            LayerType=SimpleNamespace(CONVOLUTION="conv", CONSTANT="constant", IDENTITY="identity", SHUFFLE="shuffle"))
        policy = {"fingerprint_sha256": policy_fingerprint(), "layers": [
            {"name": name, "compute_precision": "int8"} for name in manifest["int8_convolution_names"]]}
        report = apply_calibrated_ranges(network, trt, path=self.output,
            model_sha256=digest(self.destination.read_bytes()), precision_policy=policy)
        self.assertEqual(report["covered_int8_convolutions"], 462)
        self.assertEqual(report["required_conv_edge_count"], 924)
        self.assertEqual(tensors["conv_461"].dynamic_range, (-1.27, 1.27))
        with self.assertRaisesRegex(ValueError, "does not match destination"):
            apply_calibrated_ranges(network, trt, path=self.output,
                model_sha256=digest(self.source.read_bytes()), precision_policy=policy)

    def test_unrelated_change_rejected_and_no_output_created(self):
        destination = onnx.load(self.destination)
        destination.graph.node[1].input[0] = "images"
        onnx.save(destination, self.destination)
        with self.assertRaises(ValueError):
            self.rebind()
        self.assertFalse(self.output.exists())

    def test_model_cache_and_checksum_binding_rejections(self):
        original = deepcopy(self.artifact)
        for mutate in (lambda artifact: artifact.__setitem__("source_full_onnx_sha256", "0" * 64),
                       lambda artifact: artifact["source_cache"]["binding"].__setitem__("model_sha256", "0" * 64),
                       lambda artifact: artifact["source_manifest"]["document"].__setitem__("source_full_onnx_sha256", "0" * 64)):
            self.artifact = deepcopy(original)
            mutate(self.artifact)
            self.save_ranges()
            with self.assertRaises(ValueError):
                self.rebind()
            self.assertFalse(self.output.exists())
        self.artifact = deepcopy(original)
        self.artifact["artifact_sha256"] = "0" * 64
        self.ranges.write_text(json.dumps(self.artifact), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "checksum"):
            self.rebind()

    def test_msda_dependent_calibrated_tensor_rejected(self):
        self.artifact["source_manifest"]["document"]["tensor_mapping"].append(
            {"source_name": "result", "subgraph_name": "result", "dtype": "float32"})
        self.save_ranges()
        with self.assertRaisesRegex(ValueError, "depends on MSDA"):
            self.rebind()


class PluginLoadingTests(unittest.TestCase):
    def test_global_loading_retains_handle_and_reports_actual_binary_sha(self):
        with tempfile.TemporaryDirectory() as directory:
            library = Path(directory) / "plugin.so"
            library.write_bytes(b"fixture library, not executable")
            handle = object()
            with patch.dict(engine._PLUGIN_LIBRARIES, {}, clear=True), patch.object(ctypes, "CDLL", return_value=handle) as loader:
                records = engine.load_plugin_libraries([library, library])
                loader.assert_called_once_with(str(library.resolve()), mode=ctypes.RTLD_GLOBAL)
                self.assertIs(engine._PLUGIN_LIBRARIES[str(library.resolve())][0], handle)
                self.assertEqual(records, [{"path": str(library.resolve()), "sha256": digest(library.read_bytes())}])
                library.write_bytes(b"replacement")
                with self.assertRaisesRegex(RuntimeError, "changed after loading"):
                    engine.load_plugin_libraries([library])

    def test_initialize_old_args_and_load_before_use(self):
        logger = SimpleNamespace(INFO=1, WARNING=2)
        fake_trt = SimpleNamespace(__version__="10.13.3.9", Logger=lambda _: logger,
                                   init_libnvinfer_plugins=lambda *args: True)
        fake_trt.Logger = unittest.mock.Mock(return_value=logger, INFO=1, WARNING=2)
        with patch.dict(sys.modules, {"tensorrt": fake_trt}), patch.object(engine, "load_cuda_math_libraries"), \
                patch.object(engine, "load_plugin_libraries") as loader:
            engine.initialize_trt(SimpleNamespace(verbose=False))
            loader.assert_called_once_with(())
            loader.reset_mock()
            engine.initialize_trt(SimpleNamespace(verbose=False, plugin_library=["custom.so"]))
            loader.assert_called_once_with(["custom.so"])

    def test_help_requires_no_third_party_packages(self):
        source = str(Path(__file__).resolve().parents[1] / "src")
        for command in ("msda-rewrite", "msda-rebind-ranges", "build", "run", "profile"):
            with self.subTest(command=command), tempfile.TemporaryDirectory() as directory:
                result = subprocess.run([sys.executable, "-S", "-m", "bevformer_trt", command, "--help"],
                    cwd=directory, env=dict(os.environ, PYTHONPATH=source), capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("--plugin-library" if command in ("build", "run", "profile") else "--onnx", result.stdout)


if __name__ == "__main__":
    unittest.main()
