"""Generic ONNX and torch.export frontend coverage."""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import hashlib
import importlib.util
import io
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from gac.cli import main
from gac.frontends.model_frontend import ModelFrontendError, detect_model_format, load_model
from gac.frontends.torch_export_frontend import (
    TorchExportFrontendError,
    import_exported_program,
    load_exported_program,
)
from gac.runtime import reference_run


HAS_ONNX = importlib.util.find_spec("onnx") is not None


class ModelFrontendDispatchTests(unittest.TestCase):
    def test_suffix_detection_is_explicit(self):
        self.assertEqual(detect_model_format("model.json"), "json")
        self.assertEqual(detect_model_format("model.onnx"), "onnx")
        self.assertEqual(detect_model_format("model.pt2"), "torch-export")
        with self.assertRaisesRegex(ModelFrontendError, "cannot infer"):
            detect_model_format("checkpoint.pth")

    def test_json_path_does_not_import_frameworks(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "identity.json"
            path.write_text(
                """{
                  "name": "identity",
                  "inputs": {"x": {"shape": [2, 3], "dtype": "float32"}},
                  "nodes": [{"id": "copy", "op": "Identity", "inputs": ["x"], "outputs": ["y"]}],
                  "outputs": ["y"]
                }""",
                encoding="utf-8",
            )
            before = set(__import__("sys").modules)
            graph = load_model(path)
            loaded = set(__import__("sys").modules) - before
            self.assertEqual(graph.name, "identity")
            self.assertFalse(any(name == "onnx" or name.startswith("onnx.") for name in loaded))
            self.assertFalse(any(name == "torch" or name.startswith("torch.") for name in loaded))


@unittest.skipUnless(HAS_ONNX, "optional ONNX dependency is not installed")
class OnnxFrontendTests(unittest.TestCase):
    @staticmethod
    def _mlp_model(*, symbolic: bool = False):
        import onnx
        from onnx import TensorProto, helper, numpy_helper

        batch = "batch" if symbolic else 2
        x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [batch, 4])
        y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [batch, 3])
        weight = np.arange(12, dtype=np.float32).reshape(4, 3) / 10
        bias = np.asarray([-1.0, 0.5, 2.0], dtype=np.float32)
        nodes = [
            helper.make_node("MatMul", ["x", "weight"], ["product"], name="matmul"),
            helper.make_node("Add", ["product", "bias"], ["sum"], name="bias_add"),
            helper.make_node("Relu", ["sum"], ["y"], name="activation"),
        ]
        graph = helper.make_graph(
            nodes,
            "onnx_mlp",
            [x],
            [y],
            [
                numpy_helper.from_array(weight, name="weight"),
                numpy_helper.from_array(bias, name="bias"),
            ],
        )
        return helper.make_model(
            graph,
            producer_name="gac-test",
            opset_imports=[helper.make_opsetid("", 18)],
        )

    def test_mlp_import_preserves_shapes_constants_and_numerics(self):
        from gac.frontends.onnx_frontend import import_onnx_model

        graph = import_onnx_model(self._mlp_model())
        self.assertEqual([node.op for node in graph.nodes], ["MatMul", "Add", "ReLU"])
        self.assertEqual(set(graph.inputs), {"x"})
        self.assertEqual(set(graph.constants), {"weight", "bias"})
        self.assertEqual(graph.value_specs["y"].shape, (2, 3))
        self.assertEqual(graph.metadata["import"]["format"], "onnx")
        value = np.arange(8, dtype=np.float32).reshape(2, 4) / 7
        actual = reference_run(graph, {"x": value})["y"]
        expected = np.maximum(value @ graph.constants["weight"] + graph.constants["bias"], 0)
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)

    def test_symbolic_dimension_is_retained(self):
        from gac.frontends.onnx_frontend import import_onnx_model

        graph = import_onnx_model(self._mlp_model(symbolic=True))
        self.assertEqual(graph.inputs["x"].shape, ("batch", 4))
        self.assertEqual(graph.value_specs["y"].shape, ("batch", 3))

    def test_constant_shape_operand_is_legalized_to_reshape_attribute(self):
        from onnx import TensorProto, helper, numpy_helper
        from gac.frontends.onnx_frontend import import_onnx_model

        x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3])
        y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [3, 2])
        shape = numpy_helper.from_array(np.asarray([3, 2], dtype=np.int64))
        model = helper.make_model(
            helper.make_graph(
                [
                    helper.make_node("Constant", [], ["shape"], value=shape, name="shape"),
                    helper.make_node("Reshape", ["x", "shape"], ["y"], name="reshape"),
                ],
                "reshape",
                [x],
                [y],
            ),
            opset_imports=[helper.make_opsetid("", 18)],
        )
        graph = import_onnx_model(model)
        self.assertEqual(len(graph.nodes), 1)
        self.assertEqual(graph.nodes[0].op, "Reshape")
        self.assertEqual(graph.nodes[0].inputs, ("x",))
        self.assertEqual(graph.nodes[0].attrs["shape"], [3, 2])
        self.assertEqual(graph.constants["shape"].dtype, np.dtype("int64"))

    def test_scalar_initializers_and_constants_preserve_rank_and_roundtrip(self):
        from onnx import TensorProto, helper, numpy_helper
        from gac.core.ir import Graph
        from gac.frontends.onnx_frontend import import_onnx_model

        # Each encoding occurs in framework exports. A scalar is shape (),
        # distinct from a one-element vector even when both hold one number.
        for dtype, onnx_type, value_attribute in (
            (np.float32, TensorProto.FLOAT, "value_float"),
            (np.int64, TensorProto.INT64, "value_int"),
        ):
            for encoding in ("initializer", "tensor_attribute", "scalar_attribute"):
                with self.subTest(dtype=dtype, encoding=encoding):
                    value = np.asarray(3, dtype=dtype)
                    tensor = numpy_helper.from_array(value, name="scalar")
                    initializers = [tensor] if encoding == "initializer" else []
                    nodes = []
                    if encoding != "initializer":
                        attributes = (
                            {"value": tensor} if encoding == "tensor_attribute"
                            else {value_attribute: value.item()}
                        )
                        nodes.append(helper.make_node("Constant", [], ["scalar"], name="constant", **attributes))
                    nodes.append(helper.make_node("Add", ["x", "scalar"], ["y"], name="add"))
                    model = helper.make_model(
                        helper.make_graph(
                            nodes, "scalar_add",
                            [helper.make_tensor_value_info("x", onnx_type, [])],
                            [helper.make_tensor_value_info("y", onnx_type, [])],
                            initializers,
                        ),
                        opset_imports=[helper.make_opsetid("", 18)],
                    )
                    graph = import_onnx_model(model)
                    self.assertEqual(graph.constants["scalar"].shape, ())
                    self.assertEqual(graph.value_specs["scalar"].shape, ())
                    self.assertEqual(graph.value_specs["y"].shape, ())
                    self.assertEqual(graph.constants["scalar"].dtype, np.dtype(dtype))
                    inline = Graph.from_json(graph.to_json())
                    with tempfile.TemporaryDirectory() as directory:
                        archive = Path(directory) / "constants.npz"
                        np.savez(archive, **graph.constants)
                        with np.load(archive, allow_pickle=False) as arrays:
                            restored = Graph.from_dict(graph.to_dict(include_constant_data=False), external_constants=arrays)
                    for result in (inline, restored):
                        self.assertEqual(result.constants["scalar"].shape, ())
                        result.validate()
                        actual = reference_run(result, {"x": np.asarray(2, dtype=dtype)})["y"]
                        self.assertEqual(actual.shape, ())
                        np.testing.assert_array_equal(actual, np.asarray(5, dtype=dtype))

    def test_reshape_zero_copies_matching_input_axis_across_rank_change(self):
        from onnx import TensorProto, helper, numpy_helper
        from gac.frontends.onnx_frontend import import_onnx_model

        x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3, 4])
        y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 12])
        shape = numpy_helper.from_array(np.asarray([0, -1], dtype=np.int64), name="shape")
        model = helper.make_model(
            helper.make_graph(
                [helper.make_node("Reshape", ["x", "shape"], ["y"], name="reshape")],
                "reshape_zero",
                [x],
                [y],
                [shape],
            ),
            opset_imports=[helper.make_opsetid("", 18)],
        )
        graph = import_onnx_model(model)
        self.assertEqual(graph.nodes[0].attrs["shape"], [2, -1])
        self.assertEqual(graph.value_specs["y"].shape, (2, 12))

    def test_onnx_layer_norm_rejects_unmodeled_fp16_accumulator_semantics(self):
        from onnx import TensorProto, helper, numpy_helper
        from gac.frontends.onnx_frontend import OnnxFrontendError, import_onnx_model

        x = helper.make_tensor_value_info("x", TensorProto.FLOAT16, [2, 4])
        y = helper.make_tensor_value_info("y", TensorProto.FLOAT16, [2, 4])
        scale = numpy_helper.from_array(np.ones(4, dtype=np.float16), name="scale")
        model = helper.make_model(
            helper.make_graph(
                [
                    helper.make_node(
                        "LayerNormalization", ["x", "scale"], ["y"], name="norm"
                    )
                ],
                "fp16_layer_norm",
                [x],
                [y],
                [scale],
            ),
            opset_imports=[helper.make_opsetid("", 18)],
        )
        with self.assertRaisesRegex(OnnxFrontendError, "requires float32"):
            import_onnx_model(model)

    def test_optional_input_slots_are_preserved_for_clip_and_slice(self):
        from onnx import TensorProto, helper, numpy_helper
        from gac.frontends.onnx_frontend import import_onnx_model

        x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [5])
        y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [5])
        constants = [
            numpy_helper.from_array(np.asarray(1.0, dtype=np.float32), name="maximum"),
            numpy_helper.from_array(np.asarray([4], dtype=np.int64), name="starts"),
            numpy_helper.from_array(np.asarray([-6], dtype=np.int64), name="ends"),
            numpy_helper.from_array(np.asarray([-1], dtype=np.int64), name="steps"),
        ]
        model = helper.make_model(
            helper.make_graph(
                [
                    helper.make_node(
                        "Clip", ["x", "", "maximum"], ["clipped"], name="clip_max"
                    ),
                    helper.make_node(
                        "Slice",
                        ["clipped", "starts", "ends", "", "steps"],
                        ["y"],
                        name="reverse",
                    ),
                ],
                "optional_slots",
                [x],
                [y],
                constants,
            ),
            opset_imports=[helper.make_opsetid("", 18)],
        )
        graph = import_onnx_model(model)
        self.assertEqual(graph.nodes[0].attrs, {"max": 1.0})
        self.assertNotIn("axes", graph.nodes[1].attrs)
        self.assertEqual(graph.nodes[1].attrs["steps"], [-1])
        value = np.asarray([-2.0, 0.0, 2.0, 3.0, 4.0], dtype=np.float32)
        actual = reference_run(graph, {"x": value})["y"]
        np.testing.assert_array_equal(actual, np.asarray([1.0, 1.0, 1.0, 0.0, -2.0]))

    def test_dropout_training_slot_is_not_lost_when_ratio_is_omitted(self):
        from onnx import TensorProto, helper, numpy_helper
        from gac.frontends.onnx_frontend import OnnxFrontendError, import_onnx_model

        x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2])
        y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [2])
        training = numpy_helper.from_array(np.asarray(True, dtype=np.bool_), name="training")
        model = helper.make_model(
            helper.make_graph(
                [
                    helper.make_node(
                        "Dropout", ["x", "", "training"], ["y"], name="dropout"
                    )
                ],
                "training_dropout",
                [x],
                [y],
                [training],
            ),
            opset_imports=[helper.make_opsetid("", 18)],
        )
        with self.assertRaisesRegex(OnnxFrontendError, "training mode"):
            import_onnx_model(model)

    def test_legacy_dropout_training_default_is_not_changed_to_identity(self):
        from onnx import TensorProto, helper
        from gac.frontends.onnx_frontend import OnnxFrontendError, import_onnx_model

        x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2])
        y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [2])
        model = helper.make_model(
            helper.make_graph(
                [helper.make_node("Dropout", ["x"], ["y"], name="legacy_dropout")],
                "legacy_training_dropout",
                [x],
                [y],
            ),
            opset_imports=[helper.make_opsetid("", 6)],
        )
        with self.assertRaisesRegex(OnnxFrontendError, "training semantics"):
            import_onnx_model(model)

    def test_legacy_batch_norm_training_default_is_not_imported_as_inference(self):
        from onnx import TensorProto, helper, numpy_helper
        from gac.frontends.onnx_frontend import OnnxFrontendError, import_onnx_model

        x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2, 2, 2])
        y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2, 2, 2])
        initializers = [
            numpy_helper.from_array(np.ones(2, dtype=np.float32), name="scale"),
            numpy_helper.from_array(np.zeros(2, dtype=np.float32), name="bias"),
            numpy_helper.from_array(np.zeros(2, dtype=np.float32), name="mean"),
            numpy_helper.from_array(np.ones(2, dtype=np.float32), name="variance"),
        ]
        model = helper.make_model(
            helper.make_graph(
                [
                    helper.make_node(
                        "BatchNormalization",
                        ["x", "scale", "bias", "mean", "variance"],
                        ["y"],
                        name="legacy_batch_norm",
                    )
                ],
                "legacy_training_batch_norm",
                [x],
                [y],
                initializers,
            ),
            opset_imports=[helper.make_opsetid("", 6)],
        )
        with self.assertRaisesRegex(OnnxFrontendError, "inference-only"):
            import_onnx_model(model)

    def test_legacy_nonspatial_batch_norm_is_not_changed_to_channel_norm(self):
        from onnx import TensorProto, helper, numpy_helper
        from gac.frontends.onnx_frontend import OnnxFrontendError, import_onnx_model

        x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2, 2, 2])
        y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2, 2, 2])
        parameters = [
            numpy_helper.from_array(np.ones(8, dtype=np.float32), name="scale"),
            numpy_helper.from_array(np.zeros(8, dtype=np.float32), name="bias"),
            numpy_helper.from_array(np.zeros(8, dtype=np.float32), name="mean"),
            numpy_helper.from_array(np.ones(8, dtype=np.float32), name="variance"),
        ]
        model = helper.make_model(
            helper.make_graph(
                [
                    helper.make_node(
                        "BatchNormalization",
                        ["x", "scale", "bias", "mean", "variance"],
                        ["y"],
                        name="legacy_nonspatial",
                        is_test=1,
                        spatial=0,
                    )
                ],
                "legacy_nonspatial_batch_norm",
                [x],
                [y],
                parameters,
            ),
            opset_imports=[helper.make_opsetid("", 6)],
        )
        with self.assertRaisesRegex(OnnxFrontendError, "inference-only"):
            import_onnx_model(model)

    def test_grid_sample_bilinear_mode_is_not_textually_corrupted(self):
        from onnx import TensorProto, helper
        from gac.frontends.onnx_frontend import import_onnx_model

        value = helper.make_tensor_value_info("value", TensorProto.FLOAT, [1, 2, 3, 4])
        grid = helper.make_tensor_value_info("grid", TensorProto.FLOAT, [1, 5, 6, 2])
        output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 2, 5, 6])
        model = helper.make_model(
            helper.make_graph(
                [
                    helper.make_node(
                        "GridSample",
                        ["value", "grid"],
                        ["output"],
                        name="sample",
                        mode="bilinear",
                    )
                ],
                "grid_sample",
                [value, grid],
                [output],
            ),
            opset_imports=[helper.make_opsetid("", 16)],
        )
        graph = import_onnx_model(model)
        self.assertEqual(graph.nodes[0].attrs["mode"], "bilinear")

    def test_overrideable_initializer_input_is_rejected_explicitly(self):
        from onnx import TensorProto, helper, numpy_helper
        from gac.frontends.onnx_frontend import OnnxFrontendError, import_onnx_model

        x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2])
        weight = helper.make_tensor_value_info("weight", TensorProto.FLOAT, [2])
        y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [2])
        model = helper.make_model(
            helper.make_graph(
                [helper.make_node("Add", ["x", "weight"], ["y"], name="add")],
                "overrideable_initializer",
                [x, weight],
                [y],
                [
                    numpy_helper.from_array(
                        np.asarray([1.0, 2.0], dtype=np.float32), name="weight"
                    )
                ],
            ),
            opset_imports=[helper.make_opsetid("", 18)],
        )
        with self.assertRaisesRegex(OnnxFrontendError, "override/default-input"):
            import_onnx_model(model)

    def test_gemm_transpose_is_decomposed_without_mutating_source_weight(self):
        from onnx import TensorProto, helper, numpy_helper
        from gac.frontends.onnx_frontend import import_onnx_model

        x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 4])
        y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 3])
        weight = np.arange(12, dtype=np.float32).reshape(3, 4)
        bias = np.ones(3, dtype=np.float32)
        model = helper.make_model(
            helper.make_graph(
                [helper.make_node("Gemm", ["x", "weight", "bias"], ["y"], name="linear", transB=1)],
                "gemm",
                [x],
                [y],
                [
                    numpy_helper.from_array(weight, name="weight"),
                    numpy_helper.from_array(bias, name="bias"),
                ],
            ),
            opset_imports=[helper.make_opsetid("", 18)],
        )
        graph = import_onnx_model(model)
        self.assertEqual([node.op for node in graph.nodes], ["MatMul", "Add"])
        np.testing.assert_array_equal(graph.constants["weight"], weight)
        transposed = graph.nodes[0].inputs[1]
        np.testing.assert_array_equal(graph.constants[transposed], weight.T)

    def test_transpose_default_perm_is_materialized_from_input_rank(self):
        from onnx import TensorProto, helper
        from gac.frontends.onnx_frontend import import_onnx_model

        x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3, 4])
        y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [4, 3, 2])
        model = helper.make_model(
            helper.make_graph(
                [helper.make_node("Transpose", ["x"], ["y"], name="transpose")],
                "transpose_default",
                [x],
                [y],
            ),
            opset_imports=[helper.make_opsetid("", 18)],
        )
        graph = import_onnx_model(model)
        self.assertEqual(graph.nodes[0].attrs["perm"], [2, 1, 0])

    def test_pre_opset_13_softmax_flatten_semantics_are_not_silently_changed(self):
        from onnx import TensorProto, helper
        from gac.frontends.onnx_frontend import OnnxFrontendError, import_onnx_model

        x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3, 4])
        y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 3, 4])
        model = helper.make_model(
            helper.make_graph(
                [helper.make_node("Softmax", ["x"], ["y"], name="softmax")],
                "legacy_softmax",
                [x],
                [y],
            ),
            opset_imports=[helper.make_opsetid("", 12)],
        )
        with self.assertRaisesRegex(OnnxFrontendError, "pre-opset-13"):
            import_onnx_model(model)

    def test_legacy_axis_aligned_elementwise_broadcast_is_rejected(self):
        from onnx import TensorProto, helper
        from gac.frontends.onnx_frontend import OnnxFrontendError, import_onnx_model

        for operation in ("Add", "Sub", "Mul", "Div"):
            with self.subTest(operation=operation):
                left = helper.make_tensor_value_info(
                    "left", TensorProto.FLOAT, [2, 3, 4, 5]
                )
                right = helper.make_tensor_value_info("right", TensorProto.FLOAT, [3, 4])
                output = helper.make_tensor_value_info(
                    "output", TensorProto.FLOAT, [2, 3, 4, 5]
                )
                model = helper.make_model(
                    helper.make_graph(
                        [
                            helper.make_node(
                                operation,
                                ["left", "right"],
                                ["output"],
                                name="legacy_binary",
                                broadcast=1,
                                axis=1,
                            )
                        ],
                        "legacy_binary_broadcast",
                        [left, right],
                        [output],
                    ),
                    opset_imports=[helper.make_opsetid("", 6)],
                )
                with self.assertRaisesRegex(OnnxFrontendError, "axis-aligned broadcasting"):
                    import_onnx_model(model)

    def test_unsupported_operator_is_aggregated_or_retained_for_analysis(self):
        from onnx import TensorProto, helper
        from gac.frontends.onnx_frontend import OnnxFrontendError, import_onnx_model

        x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1, 2, 2])
        y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 1, 4, 4])
        model = helper.make_model(
            helper.make_graph(
                [helper.make_node("Resize", ["x"], ["y"], name="upsample")],
                "unsupported",
                [x],
                [y],
            ),
            opset_imports=[helper.make_opsetid("", 18)],
        )
        with self.assertRaisesRegex(OnnxFrontendError, r"upsample.*Resize"):
            import_onnx_model(model)
        graph = import_onnx_model(model, strict=False)
        self.assertEqual(graph.nodes[0].op, "ai.onnx::Resize")
        self.assertEqual(
            graph.metadata["import"]["unsupported_nodes"],
            [{"node": "upsample", "op": "ai.onnx::Resize"}],
        )

    def test_inspection_keeps_shape_program_and_unlowered_reshape_edges(self):
        from onnx import TensorProto, helper
        from gac.frontends.onnx_frontend import OnnxFrontendError, import_onnx_model

        for batch in (2, "batch"):
            with self.subTest(batch=batch):
                model = helper.make_model(
                    helper.make_graph(
                        [
                            helper.make_node("Shape", ["x"], ["shape"], name="shape"),
                            helper.make_node("Reshape", ["x", "shape"], ["y"], name="reshape"),
                        ],
                        "shape_program",
                        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [batch, 3])],
                        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [batch, 3])],
                    ),
                    opset_imports=[helper.make_opsetid("", 18)],
                )
                with self.assertRaisesRegex(OnnxFrontendError, "compile-time constant"):
                    import_onnx_model(model)
                graph = import_onnx_model(model, strict=False)
                self.assertEqual([node.op for node in graph.nodes], ["ai.onnx::Shape", "ai.onnx::Reshape"])
                self.assertEqual(graph.nodes[1].inputs, ("x", "shape"))
                self.assertEqual(graph.nodes[1].source_nodes, ("reshape",))
                self.assertNotIn("shape", graph.constants)
                self.assertIn("compile-time constant", graph.metadata["import"]["unsupported_nodes"][1]["reason"])
                graph.validate()

    def test_inspection_preserves_known_but_unmodeled_reshape_semantics(self):
        from onnx import TensorProto, helper, numpy_helper
        from gac.frontends.onnx_frontend import OnnxFrontendError, import_onnx_model

        model = helper.make_model(
            helper.make_graph(
                [helper.make_node("Reshape", ["x", "shape"], ["y"], name="reshape", allowzero=1)],
                "allowzero",
                [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3])],
                [helper.make_tensor_value_info("y", TensorProto.FLOAT, [3, 2])],
                [numpy_helper.from_array(np.asarray([3, 2], dtype=np.int64), name="shape")],
            ),
            opset_imports=[helper.make_opsetid("", 18)],
        )
        with self.assertRaisesRegex(OnnxFrontendError, "allowzero=1"):
            import_onnx_model(model)
        graph = import_onnx_model(model, strict=False)
        self.assertEqual(graph.nodes[0].op, "ai.onnx::Reshape")
        self.assertEqual(graph.nodes[0].attrs, {"allowzero": 1})
        self.assertEqual(graph.nodes[0].inputs, ("x", "shape"))
        self.assertIn("allowzero=1", graph.metadata["import"]["unsupported_nodes"][0]["reason"])

    def test_inspection_keeps_optional_slots_for_runtime_clip_bound(self):
        from onnx import TensorProto, helper
        from gac.frontends.onnx_frontend import import_onnx_model

        model = helper.make_model(
            helper.make_graph(
                [helper.make_node("Clip", ["x", "", "maximum"], ["y"], name="clip")],
                "runtime_clip",
                [
                    helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3]),
                    helper.make_tensor_value_info("maximum", TensorProto.FLOAT, []),
                ],
                [helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 3])],
            ),
            opset_imports=[helper.make_opsetid("", 18)],
        )
        graph = import_onnx_model(model, strict=False)
        self.assertEqual(graph.nodes[0].op, "ai.onnx::Clip")
        self.assertEqual(graph.nodes[0].inputs, ("x", "maximum"))
        self.assertEqual(graph.metadata["import"]["unsupported_nodes"][0]["input_slots"], ["x", "", "maximum"])

    def test_inspection_rolls_back_a_rejected_partial_decomposition(self):
        import gac.frontends.onnx_frontend as frontend
        from onnx import TensorProto, helper, numpy_helper

        original_translate = frontend._translate_node

        def reject_after_decomposition(proto, **kwargs):
            result = original_translate(proto, **kwargs)
            if proto.op_type == "Gemm":
                raise frontend._UnsupportedOnnxSemantics("target cannot represent this Gemm variant")
            return result

        # This source Gemm transposes its weight and allocates a temporary
        # MatMul value before the hypothetical late legality check rejects it.
        model = helper.make_model(
            helper.make_graph(
                [helper.make_node("Gemm", ["x", "weight", "bias"], ["y"], name="linear", transB=1)],
                "gemm",
                [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 4])],
                [helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 3])],
                [
                    numpy_helper.from_array(np.ones((3, 4), dtype=np.float32), name="weight"),
                    numpy_helper.from_array(np.ones(3, dtype=np.float32), name="bias"),
                ],
            ),
            opset_imports=[helper.make_opsetid("", 18)],
        )
        with patch.object(frontend, "_translate_node", side_effect=reject_after_decomposition):
            graph = frontend.import_onnx_model(model, strict=False)
        self.assertEqual(graph.nodes[0].op, "ai.onnx::Gemm")
        self.assertEqual(graph.nodes[0].id, model.graph.node[0].name)
        self.assertEqual(set(graph.constants), {value.name for value in model.graph.initializer})
        self.assertEqual(graph.nodes[0].inputs, tuple(model.graph.node[0].input))
        self.assertEqual(graph.nodes[0].attrs["transB"], 1)
        graph.validate()

    def test_inspection_does_not_hide_invalid_nodes_or_nested_control_flow(self):
        from onnx import TensorProto, helper
        from gac.frontends.onnx_frontend import OnnxFrontendError, import_onnx_model

        x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3])
        y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 3])
        invalid = helper.make_model(
            helper.make_graph([helper.make_node("Reshape", ["x"], ["y"])], "invalid", [x], [y]),
            opset_imports=[helper.make_opsetid("", 18)],
        )
        with self.assertRaisesRegex(OnnxFrontendError, "invalid ONNX model"):
            import_onnx_model(invalid, strict=False)
        with self.assertRaisesRegex(OnnxFrontendError, "needs data and shape"):
            import_onnx_model(invalid, strict=False, check_model=False, infer_shapes=False)
        branch = helper.make_graph(
            [helper.make_node("Identity", ["x"], ["branch_output"])], "branch", [],
            [helper.make_tensor_value_info("branch_output", TensorProto.FLOAT, [2, 3])],
        )
        nested = helper.make_model(
            helper.make_graph(
                [helper.make_node("If", ["condition"], ["y"], then_branch=branch, else_branch=branch)],
                "nested", [x, helper.make_tensor_value_info("condition", TensorProto.BOOL, [])], [y],
            ),
            opset_imports=[helper.make_opsetid("", 18)],
        )
        with self.assertRaisesRegex(OnnxFrontendError, "nested graph"):
            import_onnx_model(nested, strict=False)

    def test_inspection_does_not_swallow_unexpected_translator_errors(self):
        from gac.frontends.onnx_frontend import import_onnx_model

        with patch("gac.frontends.onnx_frontend._translate_node", side_effect=RuntimeError("translator bug")):
            with self.assertRaisesRegex(RuntimeError, "translator bug"):
                import_onnx_model(self._mlp_model(), strict=False)

    def test_cli_import_writes_canonical_json_and_report(self):
        import onnx

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "model.onnx"
            output = Path(directory) / "model.gac.json"
            report = Path(directory) / "import-report.json"
            onnx.save(self._mlp_model(), source)
            stdout = io.StringIO()
            stderr = io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                code = main(["import", str(source), "-o", str(output), "--report", str(report)])
            self.assertEqual(code, 0, stderr.getvalue())
            result = __import__("json").loads(stdout.getvalue())
            self.assertEqual(result["source_format"], "onnx")
            self.assertEqual(result["node_count"], 3)
            self.assertTrue(output.is_file())
            self.assertEqual(result, __import__("json").loads(report.read_text(encoding="utf-8")))
            round_trip = load_model(output)
            self.assertEqual([node.op for node in round_trip.nodes], ["MatMul", "Add", "ReLU"])

            cuda_source = Path(directory) / "model.cu"
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                compile_code = main(["compile", str(source), "-o", str(cuda_source)])
            self.assertEqual(compile_code, 0)
            self.assertIn("gac_kernel_0", cuda_source.read_text(encoding="utf-8"))


class _FakeTensorValue:
    def __init__(self, shape, dtype="float32"):
        self.shape = tuple(shape)
        self.dtype = dtype


class _FxNode:
    def __init__(self, op, name, *, target=None, args=(), kwargs=None, shape=None, dtype="float32"):
        self.op = op
        self.name = name
        self.target = target
        self.args = tuple(args)
        self.kwargs = dict(kwargs or {})
        self.meta = {} if shape is None else {"val": _FakeTensorValue(shape, dtype)}


class _FakeGraph:
    def __init__(self, nodes):
        self.nodes = list(nodes)


class _FakeGraphModule:
    def __init__(self, nodes):
        self.graph = _FakeGraph(nodes)


def _input_spec(kind, name, target=None):
    return SimpleNamespace(kind=kind, arg=SimpleNamespace(name=name), target=target)


def _output_spec(kind, name):
    return SimpleNamespace(kind=kind, arg=SimpleNamespace(name=name))


class TorchExportFrontendTests(unittest.TestCase):
    @staticmethod
    def _linear_program():
        x = _FxNode("placeholder", "x", shape=(2, 4))
        weight = _FxNode("placeholder", "p_weight", shape=(3, 4))
        bias = _FxNode("placeholder", "p_bias", shape=(3,))
        linear = _FxNode(
            "call_function",
            "linear",
            target="aten.linear.default",
            args=(x, weight, bias),
            shape=(2, 3),
        )
        relu = _FxNode(
            "call_function",
            "relu",
            target="aten.relu.default",
            args=(linear,),
            shape=(2, 3),
        )
        output = _FxNode("output", "output", args=((relu,),))
        signature = SimpleNamespace(
            input_specs=[
                _input_spec("USER_INPUT", "x"),
                _input_spec("PARAMETER", "p_weight", "layer.weight"),
                _input_spec("PARAMETER", "p_bias", "layer.bias"),
            ],
            output_specs=[_output_spec("USER_OUTPUT", "relu")],
        )
        return SimpleNamespace(
            graph_module=_FakeGraphModule([x, weight, bias, linear, relu, output]),
            graph_signature=signature,
            state_dict={
                "layer.weight": np.arange(12, dtype=np.float32).reshape(3, 4) / 10,
                "layer.bias": np.asarray([-1.0, 0.5, 2.0], dtype=np.float32),
            },
            constants={},
            range_constraints={},
        )

    def test_in_memory_exported_program_lifts_state_and_lowers_linear(self):
        program = self._linear_program()
        graph = import_exported_program(program, name="torch_mlp")
        self.assertEqual(graph.name, "torch_mlp")
        self.assertEqual(set(graph.inputs), {"x"})
        self.assertEqual([node.op for node in graph.nodes], ["MatMul", "Add", "ReLU"])
        self.assertIn("p_weight", graph.constants)
        transposed_name = graph.nodes[0].inputs[1]
        np.testing.assert_array_equal(
            graph.constants[transposed_name], program.state_dict["layer.weight"].T
        )
        value = np.arange(8, dtype=np.float32).reshape(2, 4) / 8
        actual = reference_run(graph, {"x": value})["relu"]
        expected = np.maximum(
            value @ program.state_dict["layer.weight"].T
            + program.state_dict["layer.bias"],
            0,
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)

    def test_exact_gelu_is_rejected_instead_of_changed_to_tanh(self):
        x = _FxNode("placeholder", "x", shape=(2, 4))
        gelu = _FxNode(
            "call_function",
            "gelu",
            target="aten.gelu.default",
            args=(x,),
            shape=(2, 4),
        )
        output = _FxNode("output", "output", args=((gelu,),))
        program = SimpleNamespace(
            graph_module=_FakeGraphModule([x, gelu, output]),
            graph_signature=SimpleNamespace(
                input_specs=[_input_spec("USER_INPUT", "x")],
                output_specs=[_output_spec("USER_OUTPUT", "gelu")],
            ),
            state_dict={},
            constants={},
            range_constraints={},
        )
        with self.assertRaisesRegex(TorchExportFrontendError, "current GAC GELU contract is tanh"):
            import_exported_program(program)

    def test_unknown_aten_op_has_node_and_target_diagnostic(self):
        x = _FxNode("placeholder", "x", shape=(2, 4))
        power = _FxNode(
            "call_function",
            "pow_1",
            target="aten.pow.Tensor_Scalar",
            args=(x, 2),
            shape=(2, 4),
        )
        output = _FxNode("output", "output", args=((power,),))
        program = SimpleNamespace(
            graph_module=_FakeGraphModule([x, power, output]),
            graph_signature=SimpleNamespace(
                input_specs=[_input_spec("USER_INPUT", "x")],
                output_specs=[_output_spec("USER_OUTPUT", "pow_1")],
            ),
            state_dict={},
            constants={},
            range_constraints={},
        )
        with self.assertRaisesRegex(TorchExportFrontendError, r"pow_1.*aten.pow"):
            import_exported_program(program)

    def test_reduction_dtype_override_is_not_silently_ignored(self):
        x = _FxNode("placeholder", "x", shape=(2, 4))
        reduction = _FxNode(
            "call_function",
            "sum_1",
            target="aten.sum.dim_IntList",
            args=(x, [1], False, "float64"),
            shape=(2,),
            dtype="float64",
        )
        output = _FxNode("output", "output", args=((reduction,),))
        program = SimpleNamespace(
            graph_module=_FakeGraphModule([x, reduction, output]),
            graph_signature=SimpleNamespace(
                input_specs=[_input_spec("USER_INPUT", "x")],
                output_specs=[_output_spec("USER_OUTPUT", "sum_1")],
            ),
            state_dict={},
            constants={},
            range_constraints={},
        )
        with self.assertRaisesRegex(TorchExportFrontendError, "accumulation cast"):
            import_exported_program(program)

    def test_pytorch_scalar_promotion_mismatch_is_rejected(self):
        x = _FxNode("placeholder", "x", shape=(2,), dtype="int32")
        add = _FxNode(
            "call_function",
            "add",
            target="aten.add.Scalar",
            args=(x, 0.5),
            shape=(2,),
            dtype="float32",
        )
        output = _FxNode("output", "output", args=((add,),))
        program = SimpleNamespace(
            graph_module=_FakeGraphModule([x, add, output]),
            graph_signature=SimpleNamespace(
                input_specs=[_input_spec("USER_INPUT", "x")],
                output_specs=[_output_spec("USER_OUTPUT", "add")],
            ),
            state_dict={},
            constants={},
            range_constraints={},
        )
        with self.assertRaisesRegex(TorchExportFrontendError, "changed output shape or dtype"):
            import_exported_program(program)

    def test_torch_layer_norm_rejects_unmodeled_fp16_accumulator_semantics(self):
        x = _FxNode("placeholder", "x", shape=(2, 4), dtype="float16")
        norm = _FxNode(
            "call_function",
            "layer_norm",
            target="aten.layer_norm.default",
            args=(x, [4], None, None, 1e-5, True),
            shape=(2, 4),
            dtype="float16",
        )
        output = _FxNode("output", "output", args=((norm,),))
        program = SimpleNamespace(
            graph_module=_FakeGraphModule([x, norm, output]),
            graph_signature=SimpleNamespace(
                input_specs=[_input_spec("USER_INPUT", "x")],
                output_specs=[_output_spec("USER_OUTPUT", "layer_norm")],
            ),
            state_dict={},
            constants={},
            range_constraints={},
        )
        with self.assertRaisesRegex(TorchExportFrontendError, "requires float32"):
            import_exported_program(program)

    def test_pt2_requires_trust_and_hash_is_checked_before_torch_import(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt2"
            path.write_bytes(b"not a real archive")
            with self.assertRaisesRegex(TorchExportFrontendError, "trusted=True"):
                load_exported_program(path)
            with patch(
                "gac.frontends.torch_export_frontend._torch_module",
                side_effect=AssertionError("torch must not be imported"),
            ):
                with self.assertRaisesRegex(TorchExportFrontendError, "64 hexadecimal"):
                    load_exported_program(
                        path,
                        trusted=True,
                        expected_sha256="not-a-digest",
                    )
                with self.assertRaisesRegex(TorchExportFrontendError, "SHA-256 mismatch"):
                    load_exported_program(
                        path,
                        trusted=True,
                        expected_sha256="0" * 64,
                    )

            payload = path.read_bytes()
            observed: list[bytes] = []

            def fake_load(stream):
                observed.append(stream.read())
                return self._linear_program()

            fake_torch = SimpleNamespace(export=SimpleNamespace(load=fake_load))
            with patch(
                "gac.frontends.torch_export_frontend._torch_module",
                return_value=fake_torch,
            ):
                graph = load_exported_program(
                    path,
                    trusted=True,
                    expected_sha256=hashlib.sha256(payload).hexdigest(),
                )
            self.assertEqual(observed, [payload])
            self.assertEqual(graph.outputs, ("relu",))


if __name__ == "__main__":
    unittest.main()
