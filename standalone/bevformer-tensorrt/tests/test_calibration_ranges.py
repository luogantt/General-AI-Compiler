"""CPU contract checks; fake CUDA-free networks do not validate engine accuracy."""

import copy
import hashlib
import json
from pathlib import Path
import struct
import tempfile
from types import SimpleNamespace
import unittest

from bevformer_trt.calibration_ranges import (
    CACHE_HEADER, PRODUCTION_KIND, TEST_KIND, TRT_VERSION, apply_calibrated_ranges,
    convert_bound_cache,
)
from bevformer_trt.precision_policy import policy_fingerprint
from bevformer_trt.quantization import CACHE_MAGIC, CalibrationCache


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


class Tensor:
    def __init__(self, name, dtype="float32"):
        self.name, self.dtype = name, dtype
        self.is_shape_tensor = False
        self.dynamic_range = None


class Layer:
    def __init__(self, name, input_tensor, output_tensor, precision="int8", layer_type="conv"):
        self.name, self.type, self.precision = name, layer_type, precision
        self.inputs = [input_tensor, Tensor("weights")]
        self.num_inputs = len(self.inputs)
        self.outputs, self.num_outputs = [output_tensor], 1

    def get_input(self, index):
        return self.inputs[index]

    def get_output(self, index):
        return self.outputs[index]


class Network:
    def __init__(self, *layers, inputs=()):
        self.layers, self.num_layers = layers, len(layers)
        self.inputs, self.num_inputs = inputs, len(inputs)

    def get_layer(self, index):
        return self.layers[index]

    def get_input(self, index):
        return self.inputs[index]


class RangeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cache = self.root / "input.cache"
        self.manifest_path = self.root / "manifest.json"
        self.output = self.root / "ranges.json"
        self.full = "a" * 64
        self.name = "/img_backbone/layer1/layer1.0/conv1/Conv"
        self.rule = policy_fingerprint()
        self.manifest = {
            "schema_version": 1, "kind": TEST_KIND,
            "source_full_onnx_sha256": self.full, "subgraph_onnx_sha256": self.full,
            "int8_convolution_names": [self.name], "policy_rule_fingerprint_sha256": self.rule,
            "int8_activation_edges": [{"layer": self.name, "edge": "input0", "tensor": "stem_relu"},
                                      {"layer": self.name, "edge": "output0", "tensor": "body_conv"}],
            "source_full_policy_fingerprint": "b" * 64, "precision_plan_sha256": "c" * 64,
            "coverage": {"expected_int8_convolutions": 1, "retained_int8_convolutions": 1,
                         "missing": [], "extra": []},
            "preservation": {name: True for name in (
                "node_protobufs_identical", "initializer_protobufs_identical",
                "input_value_info_identical", "tensor_names_identical")},
            "inputs": [{"name": "images", "dtype": "float32", "shape": [1, 3, 8, 8]}],
            "tensor_mapping": [{"source_name": name, "subgraph_name": name, "dtype": "float32"}
                               for name in ("images", "stem_relu", "body_conv")],
        }
        self.binding = {
            "model_sha256": self.full, "policy_fingerprint": "d" * 64,
            "tensorrt": TRT_VERSION, "algorithm": "IInt8EntropyCalibrator2",
            "calibration": {"input_name": "images", "input_shape": [1, 3, 8, 8], "input_dtype": "float32",
                            "provenance": {"source_kind": "synthetic_interface_test"},
                            "ordered_samples": [{"file": "a.npy", "sha256": "e" * 64}]},
        }
        self.payload = (CACHE_HEADER + "\nimages: 3c037b66\nstem_relu: 3b5f4f61\nbody_conv: 3a82c162\n").encode()
        self.trt = SimpleNamespace(__version__=TRT_VERSION, int8="int8", float16="float16", float32="float32",
                                   LayerType=SimpleNamespace(CONVOLUTION="conv", CONSTANT="constant",
                                       IDENTITY="identity", SHUFFLE="shuffle"))
        self.policy = {"fingerprint_sha256": self.rule, "layers": [{"name": self.name, "compute_precision": "int8"}]}
        self.body = Layer(self.name, Tensor("stem_relu"), Tensor("body_conv"))
        self.head = Layer("/pts_bbox_head/Conv", self.body.outputs[0], Tensor("head_output"), "float16")
        self.network = Network(self.body, self.head)

    def write_inputs(self):
        self.manifest_path.write_text(json.dumps(self.manifest), encoding="utf-8")
        # Each subcase creates a fresh cache; production converter never overwrites it.
        if self.cache.exists():
            self.cache.unlink()
        CalibrationCache(self.cache, self.binding).write(self.payload)

    def convert(self, **changes):
        self.write_inputs()
        options = dict(cache=self.cache, subgraph_manifest=self.manifest_path,
                       allow_synthetic_for_testing=True)
        options.update(changes)
        return convert_bound_cache(**options)

    def apply(self, **changes):
        options = dict(path=self.output, model_sha256=self.full, precision_policy=self.policy,
                       allow_synthetic_for_testing=True)
        options.update(changes)
        return apply_calibrated_ranges(self.network, self.trt, **options)

    def test_actual_format_scale_and_independent_provenance(self):
        artifact = self.convert(output=self.output)
        expected = struct.unpack("!f", bytes.fromhex("3b5f4f61"))[0]
        self.assertEqual(artifact["ranges"]["stem_relu"], {"scale": expected, "max_abs": expected * 127})
        self.assertEqual(artifact["source_cache"]["sha256"], hashlib.sha256(self.cache.read_bytes()).hexdigest())
        self.assertEqual(artifact["source_manifest"]["sha256"], hashlib.sha256(self.manifest_path.read_bytes()).hexdigest())
        self.assertIn("not_full_model", artifact["status"])
        self.assertNotIn(CACHE_MAGIC.decode(), self.output.read_text())

    def test_required_conv_edges_change_without_unknown_head_or_weights(self):
        self.convert(output=self.output)
        report = self.apply()
        self.assertEqual(report["applied_tensor_count"], 2)
        self.assertEqual(report["covered_int8_convolutions"], 1)
        self.assertIsNotNone(self.body.inputs[0].dynamic_range)
        self.assertIsNotNone(self.body.outputs[0].dynamic_range)
        self.assertIsNone(self.body.inputs[1].dynamic_range)
        self.assertIsNone(self.head.outputs[0].dynamic_range)
        self.assertEqual(self.head.precision, "float16")

    def residual_network(self):
        """Measured Conv + residual + ReLU boundaries, plus excluded weight aliases."""
        skip = Layer("/img_backbone/skip/Conv", self.body.inputs[0], Tensor("skip_activation"), "float16")
        add = Layer("/img_backbone/Add", self.body.outputs[0], Tensor("residual_add"), "float16", "elementwise")
        add.inputs[1] = skip.outputs[0]
        relu = Layer("/img_backbone/Relu", add.outputs[0], Tensor("residual_relu"), "float16", "activation")
        self.head.inputs[0] = relu.outputs[0]
        weights = Layer("weight_constant", None, self.body.inputs[1], "float32", "constant")
        alias = Layer("weight_alias", weights.outputs[0], Tensor("weight_alias"), "float32", "identity")
        view = Layer("weight_view", alias.outputs[0], Tensor("weight_view"), "float32", "shuffle")
        shape = Layer("shape_value", None, Tensor("shape_value"), "float32", "shape")
        shape.outputs[0].is_shape_tensor = True
        for layer in (skip, add, relu, weights, alias, view, shape):
            name = layer.outputs[0].name
            self.manifest["tensor_mapping"].append({"source_name": name, "subgraph_name": name,
                "dtype": "float32", "producer_name": "weight" if layer is weights else layer.name})
            self.payload += f"{name}: 3b800000\n".encode()
        # Actual but unlisted head values must not acquire backbone ranges.
        self.payload += b"head_output: 3b800000\n"
        images = Tensor("images")
        self.network = Network(self.body, skip, add, relu, self.head, weights, alias, view, shape, inputs=[images])
        return skip, add, relu, weights, alias, view, shape, images

    def test_measured_residual_fusion_boundaries_apply_without_changing_precision(self):
        skip, add, relu, weights, alias, view, shape, images = self.residual_network()
        self.convert(output=self.output)
        report = self.apply()
        self.assertEqual(report["required_conv_edge_count"], 2)
        self.assertEqual(report["required_conv_tensor_count"], 2)
        self.assertEqual(report["additional_activation_count"], 4)
        self.assertEqual(report["applied_tensor_count"], 6)
        self.assertEqual({row["tensor"] for row in report["additional_activations"]},
                         {"skip_activation", "residual_add", "residual_relu", "images"})
        for layer in (skip, add, relu):
            self.assertEqual(layer.outputs[0].dynamic_range, (-127 / 256, 127 / 256))
            self.assertEqual(layer.precision, "float16")
        for layer in (weights, alias, view, shape, self.head):
            self.assertIsNone(layer.outputs[0].dynamic_range)
        self.assertIsNotNone(images.dynamic_range)
        self.assertEqual(self.body.precision, "int8")

    def test_invalid_additional_range_fails_before_any_mutation(self):
        self.residual_network()
        self.convert(output=self.output)
        artifact = json.loads(self.output.read_text())
        artifact["ranges"]["residual_add"] = {"scale": -1, "max_abs": -127}
        artifact.pop("artifact_sha256")
        artifact["artifact_sha256"] = hashlib.sha256(canonical(artifact)).hexdigest()
        self.output.write_text(json.dumps(artifact))
        with self.assertRaisesRegex(ValueError, "Invalid measured.*residual_add"):
            self.apply()
        for layer in self.network.layers:
            self.assertIsNone(layer.outputs[0].dynamic_range)
        self.assertIsNone(self.body.inputs[0].dynamic_range)

    def test_required_edge_cannot_be_reinterpreted_as_constant_weight(self):
        self.manifest["tensor_mapping"][1]["producer_name"] = "weight"
        self.convert(output=self.output)
        with self.assertRaisesRegex(ValueError, "resolves to constant weights"):
            self.apply()
        self.assertIsNone(self.body.inputs[0].dynamic_range)

    def test_missing_required_range_fails_before_mutation(self):
        # Older manifests cannot preflight graph edges; the actual network still must.
        self.manifest.pop("int8_activation_edges")
        self.payload = (CACHE_HEADER + "\nstem_relu: 3b5f4f61\n").encode()
        self.convert(output=self.output)
        with self.assertRaisesRegex(ValueError, "Missing measured"):
            self.apply()
        self.assertIsNone(self.body.inputs[0].dynamic_range)

    def test_cache_header_hex_duplicate_and_invalid_scales(self):
        invalid = [b"wrong\na: 3f800000\n", (CACHE_HEADER + "\na: xyz\n").encode(),
                   (CACHE_HEADER + "\na: 123456789\n").encode(),
                   (CACHE_HEADER + "\na: 3f800000\na: 3f800000\n").encode()]
        for encoded in ("00000000", "bf800000", "7fc00000", "7f800000", "7f7fffff"):
            invalid.append((CACHE_HEADER + f"\na: {encoded}\n").encode())
        for payload in invalid:
            with self.subTest(payload=payload):
                self.payload = payload
                with self.assertRaises(ValueError):
                    self.convert()

    def test_unused_zero_infinity_are_audited_and_never_applied(self):
        for name in ("zero_constant", "infinite_mask"):
            self.manifest["tensor_mapping"].append(
                {"source_name": name, "subgraph_name": name, "dtype": "float32"})
        self.payload += b"zero_constant: 0\ninfinite_mask: 7f800000\ngenerated_constant: 0\n"
        artifact = self.convert(output=self.output)
        excluded = {row["tensor"]: row for row in artifact["excluded_invalid_scales"]}
        self.assertEqual(set(excluded), {"zero_constant", "infinite_mask", "generated_constant"})
        self.assertEqual(excluded["zero_constant"]["raw_hex"], "0")
        self.assertEqual(excluded["infinite_mask"]["reason"], "nonfinite")
        self.assertTrue(excluded["zero_constant"]["mapped_to_source"])
        self.assertFalse(excluded["generated_constant"]["mapped_to_source"])
        self.assertTrue(all(row["required_edge_check"] == "excluded_nonrequired" for row in excluded.values()))
        self.assertFalse(set(excluded) & set(artifact["ranges"]))
        report = self.apply()
        self.assertEqual(report["applied_tensor_count"], 2)
        self.assertEqual(report["excluded_invalid_scale_count"], 3)
        json.dumps(artifact, allow_nan=False)  # Inf is retained as hex/reason, never illegal JSON numbers.

    def test_required_zero_infinity_nan_negative_overflow_fail_conversion(self):
        original = self.payload
        for encoded in (b"0", b"7f800000", b"7fc00000", b"bf800000", b"7f7fffff"):
            with self.subTest(encoded=encoded):
                self.payload = original.replace(b"body_conv: 3a82c162", b"body_conv: " + encoded)
                with self.assertRaisesRegex(ValueError, "Invalid required dynamic range"):
                    self.convert(output=self.output)
                self.assertFalse(self.output.exists())

    def test_legacy_manifest_invalid_required_fails_application_before_mutation(self):
        self.manifest.pop("int8_activation_edges")
        self.payload = self.payload.replace(b"body_conv: 3a82c162", b"body_conv: 0")
        artifact = self.convert(output=self.output)
        self.assertEqual(artifact["required_edges_validation"], "deferred_to_destination_network")
        with self.assertRaisesRegex(ValueError, "Invalid required dynamic range"):
            self.apply()
        self.assertIsNone(self.body.inputs[0].dynamic_range)

    def test_required_edge_manifest_is_complete_and_matches_network(self):
        original = copy.deepcopy(self.manifest)
        self.manifest["int8_activation_edges"].pop()
        with self.assertRaisesRegex(ValueError, "Incomplete required"):
            self.convert()
        self.manifest = copy.deepcopy(original)
        self.manifest["int8_activation_edges"].append(self.manifest["int8_activation_edges"][0])
        with self.assertRaisesRegex(ValueError, "duplicate required"):
            self.convert()
        self.manifest = copy.deepcopy(original)
        self.payload = self.payload.replace(b"body_conv: 3a82c162\n", b"")
        with self.assertRaisesRegex(ValueError, "Missing required"):
            self.convert()
        self.payload += b"body_conv: 3a82c162\n"
        self.manifest["int8_activation_edges"][0]["tensor"] = "images"
        self.convert(output=self.output)
        with self.assertRaisesRegex(ValueError, "differs from destination"):
            self.apply()
        self.assertIsNone(self.body.inputs[0].dynamic_range)

    def test_short_hex_is_zero_extended_and_invalid_names_cannot_repeat(self):
        self.payload = self.payload.replace(b"images: 3c037b66", b"images: ff")
        artifact = self.convert()
        self.assertEqual(artifact["ranges"]["images"]["scale"], struct.unpack("!f", bytes.fromhex("000000ff"))[0])
        self.payload += b"unused: 0\nunused: 7f800000\n"
        with self.assertRaisesRegex(ValueError, "duplicate tensor"):
            self.convert()

    def test_cache_binding_payload_and_source_hash_checked(self):
        for field, value in (("tensorrt", "10.13.3"), ("algorithm", "MinMax"), ("model_sha256", "f" * 64)):
            with self.subTest(field=field):
                previous = self.binding[field]
                self.binding[field] = value
                with self.assertRaises(ValueError):
                    self.convert()
                self.binding[field] = previous
        self.write_inputs()
        self.cache.write_bytes(self.cache.read_bytes() + b"tampered")
        with self.assertRaisesRegex(ValueError, "checksum"):
            convert_bound_cache(cache=self.cache, subgraph_manifest=self.manifest_path, allow_synthetic_for_testing=True)

    def test_production_coverage_and_identity_opt_in(self):
        with self.assertRaisesRegex(ValueError, "opt-in"):
            self.convert(allow_synthetic_for_testing=False)
        self.manifest["kind"] = PRODUCTION_KIND
        with self.assertRaisesRegex(ValueError, "462"):
            self.convert()

    def test_preservation_mapping_and_calibration_provenance_checked(self):
        original = copy.deepcopy(self.manifest)
        for key in self.manifest["preservation"]:
            self.manifest = copy.deepcopy(original)
            self.manifest["preservation"][key] = False
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "preservation"):
                self.convert()
        self.manifest = original
        self.manifest["tensor_mapping"][0]["subgraph_name"] = "renamed"
        with self.assertRaisesRegex(ValueError, "mapping"):
            self.convert()
        self.manifest["tensor_mapping"][0]["subgraph_name"] = "images"
        self.binding["calibration"]["provenance"]["source_kind"] = "random"
        with self.assertRaisesRegex(ValueError, "source_kind"):
            self.convert()

    def test_destination_hash_version_policy_and_coverage_checked(self):
        self.convert(output=self.output)
        with self.assertRaisesRegex(ValueError, "destination"):
            self.apply(model_sha256="f" * 64)
        self.trt.__version__ = "10.13.2"
        with self.assertRaisesRegex(ValueError, "exactly"):
            self.apply()
        self.trt.__version__ = TRT_VERSION
        policy = copy.deepcopy(self.policy)
        policy["fingerprint_sha256"] = "a" * 64
        with self.assertRaisesRegex(ValueError, "policy"):
            self.apply(precision_policy=policy)
        policy = copy.deepcopy(self.policy)
        policy["layers"].append({"name": "/unknown", "compute_precision": "int8"})
        with self.assertRaisesRegex(ValueError, "coverage"):
            self.apply(precision_policy=policy)

    def test_artifact_tampering_and_destination_precision_checked(self):
        self.convert(output=self.output)
        self.body.precision = "float16"
        with self.assertRaisesRegex(ValueError, "explicitly INT8"):
            self.apply()
        self.body.precision = "int8"
        data = json.loads(self.output.read_text())
        data["ranges"]["stem_relu"]["max_abs"] = 100.0
        self.output.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "checksum"):
            self.apply()

    def test_no_overwrite_and_unmapped_cache_not_guessed(self):
        self.payload += b"extra-generated-tensor: 3f800000\n"
        artifact = self.convert(output=self.output)
        self.assertEqual(artifact["unmapped_cache_tensors"], ["extra-generated-tensor"])
        self.assertNotIn("extra-generated-tensor", artifact["ranges"])
        with self.assertRaises(FileExistsError):
            self.convert(output=self.output)


if __name__ == "__main__":
    unittest.main(verbosity=2)
