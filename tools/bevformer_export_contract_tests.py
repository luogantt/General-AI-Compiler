"""Export real custom operator call sites, independent of hand-written ONNX.

The fixtures check the exporter/adapter boundary. Native compiler tests can
consume model.onnx, inputs.npz and expected.npz without importing PyTorch.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import onnx
import torch

from export_bevformer_onnx import _cpu_extension_imports, _load_checkpoint, build_original_model
from bevformer_export_ops import MSDeformAttnExport, ModulatedDeformConvExport, install_dcn_export, validate_export_operator_contracts


def generate(output: Path) -> dict:
    _cpu_extension_imports()
    from mmcv.ops.multi_scale_deform_attn import multi_scale_deformable_attn_pytorch
    from torchvision.ops import deform_conv2d

    torch.set_num_threads(1)
    generator = torch.Generator().manual_seed(901)
    output.mkdir(parents=True, exist_ok=True)
    report = {"reference_contracts": validate_export_operator_contracts(), "cases": []}

    class FailingOracleProbe(torch.nn.Module):
        def forward(self, value):
            # Exercise the actual private tracing-state API and finally block.
            # This intentionally failing test oracle is never used for export.
            state = torch._C._get_tracing_state()
            if state is None:
                raise AssertionError("The probe must execute inside a real JIT trace")
            with patch("bevformer_export_ops.streaming_ms_deform_attn",
                       side_effect=RuntimeError("intentional oracle failure")):
                try:
                    MSDeformAttnExport.forward(None, value, None, None, None)
                except RuntimeError as error:
                    if str(error) != "intentional oracle failure":
                        raise
                else:
                    raise AssertionError("The test oracle should have raised")
            if torch._C._get_tracing_state() is not state:
                raise AssertionError("Opaque oracle failed to restore active trace on exception")
            return value + 1

    probe_input = torch.ones(1)
    probe = torch.jit.trace(FailingOracleProbe(), probe_input, check_trace=False)
    torch.testing.assert_close(probe(probe_input), probe_input + 1, rtol=0, atol=0)
    if torch._C._get_tracing_state() is not None:
        raise AssertionError("The probe leaked tracing state into normal eager execution")
    report["opaque_trace_exception_restored"] = True

    def save(name, module, inputs, expected, domain, op_type):
        directory = output / name
        directory.mkdir(exist_ok=True)
        with torch.no_grad():
            actual = module(*inputs.values())
            torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
            traced = []
            hook = module.register_forward_hook(lambda _module, _inputs, value: traced.append(value))
            try:
                torch.onnx.export(module.eval(), tuple(inputs.values()), str(directory / "model.onnx"),
                                  input_names=list(inputs), output_names=["Y"], opset_version=17,
                                  custom_opsets={domain: 1}, do_constant_folding=True)
            finally:
                hook.remove()
            if len(traced) != 1:
                raise AssertionError("Expected exactly one captured real tracing forward")
            torch.testing.assert_close(traced[0], actual, rtol=0, atol=0)
        model = onnx.load(directory / "model.onnx")
        onnx.checker.check_model(model, full_check=True)
        custom = [node for node in model.graph.node if node.domain]
        if len(custom) != 1 or (custom[0].domain, custom[0].op_type) != (domain, op_type):
            raise AssertionError("The original custom-op call must survive ONNX export exactly once")
        if any(node.op_type in {"GridSample", "Loop"} for node in model.graph.node):
            raise AssertionError("The eager reference must not be expanded into the exported custom op")
        np.savez(directory / "inputs.npz", **{key: value.numpy() for key, value in inputs.items()})
        np.savez(directory / "expected.npz", Y=expected.numpy())
        attrs = {attr.name: onnx.helper.get_attribute_value(attr) for attr in custom[0].attribute}
        attrs = {key: value.decode() if isinstance(value, bytes) else value for key, value in attrs.items()}
        report["cases"].append({"name": name, "valid": True, "kind": op_type,
                                "domain": domain, "attributes": attrs,
                                "nodes": len(model.graph.node), "shape": list(expected.shape),
                                "trace_reference_exact": True,
                                "max_abs_error": float((actual - expected).abs().max())})

    class Deform(torch.nn.Module):
        def __init__(self, weight, bias):
            super().__init__()
            self.register_buffer("weight", weight)
            self.register_buffer("bias", bias)

        def forward(self, x, offset, mask):
            return ModulatedDeformConvExport.apply(
                x, offset, mask, self.weight, self.bias, (2, 1), (1, 2), (1, 2), 2, 2)

    x = torch.randn(1, 4, 5, 7, generator=generator)
    offset = torch.randn(1, 36, 3, 7, generator=generator) * .7
    mask = torch.rand(1, 18, 3, 7, generator=generator)
    weight = torch.randn(4, 2, 3, 3, generator=generator)
    bias = torch.randn(4, generator=generator)
    expected = deform_conv2d(x, offset, weight, bias, stride=(2, 1), padding=(1, 2),
                             dilation=(1, 2), mask=mask)
    save("upstream_dcn_symbolic", Deform(weight, bias),
         {"X": x, "offset": offset, "mask": mask}, expected,
         "mmcv", "MMCVModulatedDeformConv2d")

    class Attention(torch.nn.Module):
        def __init__(self, shapes):
            super().__init__()
            self.register_buffer("spatial_shapes", shapes)

        def forward(self, value, locations, weights):
            return MSDeformAttnExport.apply(value, self.spatial_shapes, locations, weights)

    MSDeformAttnExport.query_chunk = 3
    for levels in ([(3, 4)], [(3, 4), (2, 3), (1, 2), (1, 1)]):
        shapes = torch.tensor(levels, dtype=torch.int64)
        value = torch.randn(2, sum(h * w for h, w in levels), 2, 3, generator=generator)
        locations = torch.rand(2, 11, 2, len(levels), 4, 2, generator=generator) * 1.6 - .3
        weights = torch.randn(2, 11, 2, len(levels), 4, generator=generator).softmax(-1)
        expected = multi_scale_deformable_attn_pytorch(value, shapes, locations, weights)
        name = f"upstream_msda_{len(levels)}_levels_symbolic"
        save(name, Attention(shapes), {"value": value, "locations": locations, "weights": weights},
             expected, "gac", "MSDeformAttn")
        model = onnx.load(output / name / "model.onnx")
        node = next(node for node in model.graph.node if node.domain == "gac")
        constants = {value.name: value for value in model.graph.initializer}
        if node.input[1] not in constants or constants[node.input[1]].data_type != onnx.TensorProto.INT64:
            raise AssertionError("MSDA spatial_shapes must be an explicit INT64 initializer")

    report["success"] = True
    (output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def validate_camera_batching(source_root: Path, checkpoint: Path) -> dict:
    """Compare original base backbone/FPN batching with serialized cameras.

    Small images bound this adapter test's memory. It does not claim the full
    base detector has run: the original model profile is tested separately.
    """
    config = source_root / "projects/configs/bevformer/bevformer_base.py"
    model, _, _ = build_original_model(source_root, config)
    install_dcn_export()
    weights = _load_checkpoint(model, checkpoint, trusted=True)
    images = torch.randn(2, 3, 64, 96, generator=torch.Generator().manual_seed(442))
    with torch.no_grad():
        expected = model.img_neck(model.img_backbone(images))
        chunks = [model.img_neck(model.img_backbone(image[None])) for image in images]
        actual = [torch.cat(level) for level in zip(*chunks)]
    levels = []
    for index, (value, reference) in enumerate(zip(actual, expected)):
        torch.testing.assert_close(value, reference, rtol=1e-4, atol=1e-4)
        levels.append({"level": index, "shape": list(value.shape),
                       "max_abs_error": float((value - reference).abs().max())})
    return {"passed": True, "checkpoint": weights, "rtol": 1e-4, "atol": 1e-4,
            "scope": "Two 64x96 images, original R101/DCNv2/FPN eval, batch2 vs serial batch1; not full detector",
            "levels": levels}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--base-checkpoint", type=Path,
                        help="Trusted checkpoint only; enables small original base camera parity test")
    parser.add_argument("--trusted-checkpoint", action="store_true")
    args = parser.parse_args()
    if args.base_checkpoint and (not args.source_root or not args.trusted_checkpoint):
        parser.error("--base-checkpoint requires --source-root and --trusted-checkpoint")
    report = generate(args.output)
    if args.base_checkpoint:
        report["camera_batching"] = validate_camera_batching(args.source_root, args.base_checkpoint)
        (args.output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
