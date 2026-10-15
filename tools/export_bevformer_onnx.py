"""Export the original BEVFormer PyTorch modules to ONNX for frontend inspection.

This is deliberately independent of GAC Graph construction.  The exported graph
comes from executing the upstream ResNet/FPN/BEVFormerHead modules with their
checkpoint.  Its cold-frame camera calibration and CAN-bus metadata are frozen
constants, while the image tensor remains an ONNX input.  It ends at the raw
head outputs; NMS-free box decoding and dataset preprocessing are outside it.

The optional CPU import shim only exposes unavailable MMCV extension symbols;
every such symbol raises if called.  Actual deformable attention uses MMCV's
existing, numerically equivalent PyTorch GridSample implementation.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
from pathlib import Path
import sys
import types


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _namespace(name: str, directory: Path | None = None) -> types.ModuleType:
    """Create an import namespace without executing training/dataset __init__."""
    module = types.ModuleType(name)
    module.__path__ = [] if directory is None else [str(directory)]
    sys.modules[name] = module
    if "." in name:
        parent, child = name.rsplit(".", 1)
        if parent in sys.modules:
            setattr(sys.modules[parent], child, module)
    return module


def _cpu_extension_imports() -> list[str]:
    """Permit CPU source imports, never silently emulate a compiled operator."""
    from mmcv.utils import ext_loader

    missing = []
    original = ext_loader.load_ext

    def load_ext(name, funcs):
        try:
            return original(name, funcs)
        except (ImportError, ModuleNotFoundError):
            module = types.ModuleType("mmcv." + name)
            for function in funcs:
                symbol = name + "." + function
                missing.append(symbol)

                def unavailable(*args, _symbol=symbol, **kwargs):
                    raise RuntimeError(
                        "CPU ONNX export attempted an unavailable MMCV extension: "
                        + _symbol
                    )

                setattr(module, function, unavailable)
            return module

    ext_loader.load_ext = load_ext
    return missing


def _import_upstream(source_root: Path):
    """Import real inference modules, excluding unrelated mmdet3d training code."""
    unavailable_extensions = _cpu_extension_imports()
    from mmdet.core.bbox import build_bbox_coder
    from mmdet.models import build_backbone, build_head, build_neck

    # BEVFormerHead only needs the ordinary bbox-coder registry from mmdet3d.
    # The registered NMSFreeCoder below is loaded from BEVFormer's real source.
    # No operator, attention block, projection, or numerical kernel is stubbed.
    for name in (
        "mmdet3d", "mmdet3d.core", "mmdet3d.core.bbox",
        "mmdet3d.core.bbox.coders",
    ):
        _namespace(name)
    sys.modules["mmdet3d.core.bbox.coders"].build_bbox_coder = build_bbox_coder

    directories = (
        "projects", "projects.mmdet3d_plugin",
        "projects.mmdet3d_plugin.core", "projects.mmdet3d_plugin.core.bbox",
        "projects.mmdet3d_plugin.core.bbox.coders",
        "projects.mmdet3d_plugin.models", "projects.mmdet3d_plugin.models.utils",
        "projects.mmdet3d_plugin.bevformer",
        "projects.mmdet3d_plugin.bevformer.modules",
        "projects.mmdet3d_plugin.bevformer.dense_heads",
    )
    for name in directories:
        _namespace(name, source_root.joinpath(*name.split(".")))

    names = (
        "projects.mmdet3d_plugin.core.bbox.coders.nms_free_coder",
        "projects.mmdet3d_plugin.bevformer.modules.temporal_self_attention",
        "projects.mmdet3d_plugin.bevformer.modules.spatial_cross_attention",
        "projects.mmdet3d_plugin.bevformer.modules.encoder",
        "projects.mmdet3d_plugin.bevformer.modules.decoder",
        "projects.mmdet3d_plugin.bevformer.modules.transformer",
        "projects.mmdet3d_plugin.bevformer.dense_heads.bevformer_head",
    )
    for name in names:
        importlib.import_module(name)
    return build_backbone, build_neck, build_head, unavailable_extensions


def build_original_model(source_root: Path, config_path: Path):
    """Build official inference layers with original names for strict loading."""
    import torch
    from mmcv import Config

    build_backbone, build_neck, build_head, unavailable = _import_upstream(source_root)
    config = Config.fromfile(str(config_path))
    model_config = copy.deepcopy(config.model)
    backbone_config = copy.deepcopy(model_config.img_backbone)
    backbone_config["pretrained"] = None  # Checkpoint supplies all parameters.
    head_config = copy.deepcopy(model_config.pts_bbox_head)
    head_config["train_cfg"] = None

    class OriginalColdFrame(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.img_backbone = build_backbone(backbone_config)
            self.img_neck = build_neck(model_config.img_neck)
            self.pts_bbox_head = build_head(head_config)
            self.img_metas = None

        def forward(self, images):
            # Same inference operations as BEVFormer.extract_img_feat; avoid
            # the original in-place img.squeeze_() on the public input tensor.
            batch, cameras, channels, height, width = images.shape
            features = self.img_backbone(images.reshape(batch * cameras, channels, height, width))
            if isinstance(features, dict):
                features = list(features.values())
            features = self.img_neck(features)
            features = [feature.reshape(batch, cameras, *feature.shape[1:]) for feature in features]
            result = self.pts_bbox_head(features, self.img_metas, prev_bev=None)
            return result["bev_embed"], result["all_cls_scores"], result["all_bbox_preds"]

    return OriginalColdFrame().cpu().eval(), config, unavailable


def _metadata(path: Path | None, *, height: int, width: int) -> tuple[list[dict], str]:
    import numpy as np

    if path:
        values = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(values, dict):
            values = [values]
        description = "user-provided fixed calibration and CAN-bus metadata"
    else:
        # Explicitly synthetic affine projection: every BEV cell is visible.
        # This exercises the complete network without claiming nuScenes accuracy.
        projection = [
            [width / 120.0, 0, 0, width / 2.0],
            [0, height / 120.0, 0, height / 2.0],
            [0, 0, 0, 1],
            [0, 0, 0, 1],
        ]
        values = [{
            "can_bus": [0.0] * 18,
            "lidar2img": [projection for _ in range(6)],
            "img_shape": [[height, width, 3] for _ in range(6)],
        }]
        description = "synthetic fixed all-visible projection; zero CAN-bus; not a dataset accuracy fixture"
    if len(values) != 1:
        raise ValueError("This inspectable cold-frame export currently supports batch size 1")
    for item in values:
        item["can_bus"] = np.asarray(item["can_bus"], dtype=np.float32)
        item["lidar2img"] = np.asarray(item["lidar2img"], dtype=np.float32)
        if item["can_bus"].shape != (18,) or item["lidar2img"].shape != (6, 4, 4):
            raise ValueError("Expected can_bus[18] and lidar2img[6,4,4]")
    return values, description


def _load_checkpoint(model, path: Path, trusted: bool) -> dict:
    import torch

    if not trusted:
        raise ValueError(".pth may contain Python pickle; pass --trusted-checkpoint only for a trusted file")
    checkpoint = torch.load(str(path), map_location="cpu")
    state = checkpoint.get("state_dict", checkpoint)
    if not isinstance(state, dict):
        raise ValueError("Checkpoint must contain a state_dict mapping")
    state = {name.removeprefix("module."): value for name, value in state.items()}
    # Strict loading proves every inference weight belongs to the original model.
    model.load_state_dict(state, strict=True)
    return {"state_tensors": len(state), "strict_load": True, "sha256": _sha256(path)}


def _tracer_compatibility(model) -> list[dict]:
    """Preserve arithmetic flags and reference dtypes in PyTorch 2.0 export.

    Upstream uses these configuration flags as arithmetic 0/1 multipliers.
    Replacing Python True/False by int 1/0 preserves the exact computation and
    leaves upstream source files and tensor parameters untouched.
    """
    transformer = model.pts_bbox_head.transformer
    adaptations = []
    for name in ("use_can_bus", "use_shift"):
        value = getattr(transformer, name)
        if isinstance(value, bool):
            setattr(transformer, name, int(value))
            adaptations.append({
                "attribute": "pts_bbox_head.transformer." + name,
                "before": value, "after": int(value),
                "reason": "PyTorch 2.0 ONNX/JIT Tensor-times-bool compatibility",
            })
    # The old ONNX exporter can turn linspace/meshgrid into DOUBLE constants,
    # even though get_reference_points was explicitly requested as FP32.  Keep
    # the original algorithm and make its promised output dtype explicit at
    # this boundary.  This is an eager no-op, checked against original outputs.
    import torch
    encoder = transformer.encoder
    original_reference_points = encoder.get_reference_points

    def reference_points(*args, **kwargs):
        result = original_reference_points(*args, **kwargs)
        return result.to(dtype=kwargs.get("dtype", torch.float))

    encoder.get_reference_points = reference_points
    adaptations.append({
        "attribute": "pts_bbox_head.transformer.encoder.get_reference_points",
        "reason": "Make requested dtype explicit after original linspace/meshgrid construction",
        "before": "original reference-point algorithm", "after": "same algorithm plus dtype-preserving .to(dtype)",
    })
    return adaptations


def export(args) -> dict:
    import numpy as np
    import onnx
    import torch

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    model, config, unavailable = build_original_model(args.source_root, args.config)
    checkpoint = _load_checkpoint(model, args.checkpoint, args.trusted_checkpoint)
    model.img_metas, metadata_description = _metadata(
        args.metadata, height=args.image_height, width=args.image_width,
    )
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    metadata_json = [{
        **item, "can_bus": item["can_bus"].tolist(), "lidar2img": item["lidar2img"].tolist(),
    } for item in model.img_metas]
    (output / "fixed-metadata.json").write_text(json.dumps(metadata_json, indent=2), encoding="utf-8")
    report = {
        "source": "original upstream PyTorch modules, not GAC Graph reconstruction",
        "scope": "BEVFormer-tiny complete cold raw-head path; fixed calibration/CAN-bus; no box decoding",
        "wrapper_boundary": (
            "Tensor-only wrapper repeats extract_img_feat batch/camera reshape and calls the "
            "original mmdet backbone/FPN and unmodified upstream BEVFormerHead.forward. "
            "Evaluation-time GridMask is an identity and is omitted. Dataset image "
            "normalization/padding and final box decoding are external."
        ),
        "source_root": str(args.source_root), "config": str(args.config),
        "config_sha256": _sha256(args.config),
        "checkpoint": checkpoint, "metadata": metadata_description,
        "input": {"images": [1, 6, 3, args.image_height, args.image_width]},
        "versions": {"torch": torch.__version__, "onnx": onnx.__version__},
        "opset": args.opset, "cpu_extension_symbols_guarded": sorted(set(unavailable)),
        "upstream_sources": {},
    }
    for name, module in sorted(sys.modules.items()):
        filename = getattr(module, "__file__", None)
        if name.startswith("projects.mmdet3d_plugin") and filename and Path(filename).is_file():
            source = Path(filename)
            report["upstream_sources"][str(source.relative_to(args.source_root))] = _sha256(source)
    (output / "export-report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"model_built": True, "strict_loaded_tensors": checkpoint["state_tensors"]}), flush=True)
    if args.build_only:
        return report

    images = torch.randn(1, 6, 3, args.image_height, args.image_width)
    images_path = output / "images.npy"
    if images_path.exists():
        np.testing.assert_array_equal(np.load(images_path, allow_pickle=False), images.numpy())
    output_names = ["bev_embed", "all_cls_scores", "all_bbox_preds"]
    with torch.no_grad():
        expected = model(images)
        # Existing eager fixtures from a failed exporter run must not silently
        # be replaced by a different computation.
        expected_path = output / "pytorch-outputs.npz"
        if expected_path.exists():
            with np.load(expected_path) as previous:
                for name, value in zip(output_names, expected):
                    np.testing.assert_array_equal(previous[name], value.cpu().numpy())
        np.save(images_path, images.numpy())
        np.savez(output / "pytorch-outputs.npz", **{
            name: value.detach().cpu().numpy() for name, value in zip(output_names, expected)
        })
        report["outputs"] = {name: list(value.shape) for name, value in zip(output_names, expected)}
        print(json.dumps({"forward_complete": True, "outputs": report["outputs"]}), flush=True)
        report["export_compatibility"] = _tracer_compatibility(model)
        compatible = model(images)
        report["compatibility_eager_max_abs"] = {}
        for name, original, adapted in zip(output_names, expected, compatible):
            np.testing.assert_array_equal(original.cpu().numpy(), adapted.cpu().numpy())
            report["compatibility_eager_max_abs"][name] = float((original - adapted).abs().max())
        (output / "export-report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        onnx_path = output / "bevformer-tiny-cold.onnx"
        torch.onnx.export(
            model, (images,), str(onnx_path), export_params=True,
            input_names=["images"], output_names=output_names,
            opset_version=args.opset, do_constant_folding=True,
            training=torch.onnx.TrainingMode.EVAL,
        )
    exported = onnx.load(str(onnx_path))
    onnx.checker.check_model(exported, full_check=True)
    # GridSample's schema permits separate data/grid type variables, whereas
    # the source PyTorch operator and ORT CPU implementation require matching
    # floating-point types.  Check that stronger source-model contract too.
    typed = onnx.shape_inference.infer_shapes(exported, strict_mode=True, check_type=True)
    types_by_name = {
        value.name: value.type.tensor_type.elem_type
        for value in (*typed.graph.input, *typed.graph.value_info, *typed.graph.output)
    }
    for node in typed.graph.node:
        if node.op_type == "GridSample" and types_by_name.get(node.input[0]) != types_by_name.get(node.input[1]):
            raise ValueError(f"Exporter changed GridSample data/grid dtype agreement: {node.name}")
    report["onnx"] = {
        "file": onnx_path.name, "sha256": _sha256(onnx_path),
        "nodes": len(exported.graph.node), "initializers": len(exported.graph.initializer),
        "checker_full_check": True,
        "grid_sample_dtype_agreement": True,
    }
    (output / "export-report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--trusted-checkpoint", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--image-height", type=int, default=480)
    parser.add_argument("--image-width", type=int, default=800)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--build-only", action="store_true")
    args = parser.parse_args(argv)
    args.source_root = args.source_root.resolve()
    args.config = (args.config or args.source_root / "projects/configs/bevformer/bevformer_tiny.py").resolve()
    if args.opset < 16:
        parser.error("opset >= 16 is required for the real GridSample attention implementation")
    report = export(args)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
