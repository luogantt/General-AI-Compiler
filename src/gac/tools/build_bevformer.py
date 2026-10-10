"""Offline BEVFormer artifact builder.

This command deliberately keeps framework deserialization separate from
compiler construction.  It consumes the audited JSON contract and the
data-only ``.gacweights`` format, so building an HXP never imports PyTorch or
MMCV and never executes checkpoint content.
"""

from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import hmac
import json
from pathlib import Path
import re
import sys
from typing import Any, Sequence

from gac.backends.cuda import compile_cuda
from gac.bevformer_spatial import (
    SpatialCrossAttentionConfig,
    build_spatial_cross_attention_graph,
)
from gac.bevformer_temporal import (
    TemporalSelfAttentionConfig,
    build_temporal_self_attention_graph,
)
from gac.bevformer_decoder import (
    DecoderCrossAttentionConfig,
    build_decoder_cross_attention_graph,
)
from gac.bevformer_ffn import BEVFormerFFNConfig, build_bevformer_ffn_graph
from gac.bevformer_decoder_self import (
    DecoderSelfAttentionConfig,
    build_decoder_self_attention_graph,
)
from gac.bevformer_decoder_layer import (
    BevFormerDecoderLayerConfig,
    build_bevformer_decoder_layer_graph,
)
from gac.bevformer_detection_head import (
    DetectionHeadConfig,
    build_detection_head_graph,
)
from gac.bevformer_refinement import (
    DecoderReferenceRefinementConfig,
    build_decoder_reference_refinement_graph,
)
from gac.bevformer_encoder_layer import (
    BevFormerEncoderLayerConfig,
    build_bevformer_encoder_layer_graph,
)
from gac.bevformer_encoder_stack import (
    BevFormerEncoderStackConfig,
    build_bevformer_encoder_stack_graph,
)
from gac.bevformer_query_init import (
    DecoderQueryInitConfig,
    build_decoder_query_init_graph,
)
from gac.bevformer_feature_prep import (
    BevFormerFeaturePrepConfig,
    build_bevformer_feature_prep_graph,
)
from gac.bevformer_encoder_inputs import (
    BevFormerEncoderInputConfig,
    build_bevformer_encoder_input_graph,
)
from gac.bevformer_decoder_stack import (
    BevFormerDecoderStackConfig,
    build_bevformer_decoder_stack_graph,
)
from gac.bevformer_detection_stack import (
    DetectionStackConfig,
    build_detection_stack_graph,
)
from gac.bevformer_decoder_pipeline import (
    BevFormerDecoderPipelineConfig,
    build_bevformer_decoder_pipeline_graph,
)
from gac.bevformer_cold_pipeline import (
    BevFormerColdPipelineConfig,
    build_bevformer_cold_pipeline_graph,
)
from gac.bevformer_history_pipeline import (
    BevFormerHistoryPipelineConfig,
    build_bevformer_history_pipeline_graph,
)
from gac.bevformer_step_pipeline import (
    BevFormerStepPipelineConfig,
    build_bevformer_step_pipeline_graph,
)
from gac.errors import GACError, ValidationError
from gac.frontends.bevformer_backbone import (
    BEVFORMER_R50_FPN_PROFILE,
    BevFormerBackboneError,
    ResNetFpnProfile,
    build_bevformer_backbone_graph,
    validate_checkpoint_weights,
)
from gac.frontends.bevformer_export import (
    BevFormerExportError,
    build_bevformer_tiny_manifest,
    save_bevformer_manifest,
)
from gac.frontends.bevformer_weights import (
    extract_decoder_cross_attention_weights,
    extract_decoder_layer_weights,
    extract_decoder_query_init_weights,
    extract_decoder_refinement_weights,
    extract_decoder_self_attention_weights,
    extract_decoder_stack_weights,
    extract_detection_head_weights,
    extract_detection_stack_weights,
    extract_encoder_ffn_weights,
    extract_encoder_input_weights,
    extract_encoder_layer_weights,
    extract_feature_prep_weights,
    extract_spatial_attention_weights,
    extract_temporal_attention_weights,
)
from gac.package import load_package, rebuild_cuda_program, write_package
from gac.weights import LoadedWeights, load_weights


_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _expected_digest(value: str | None) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValidationError("expected checkpoint SHA256 must contain 64 hexadecimal characters")
    return value.lower()


def _check_provenance(weights: LoadedWeights, expected: str | None) -> None:
    normalized = _expected_digest(expected)
    if normalized is not None and not hmac.compare_digest(
        normalized, weights.source_checkpoint_sha256
    ):
        raise ValidationError(
            "weights provenance mismatch: expected checkpoint "
            f"{normalized}, got {weights.source_checkpoint_sha256}"
        )


def inspect_bevformer_weights(
    path: str | Path,
    *,
    expected_checkpoint_sha256: str | None = None,
) -> dict[str, Any]:
    """Verify a data-only weight archive and return a compact inventory."""

    source = Path(path)
    loaded = load_weights(source)
    _check_provenance(loaded, expected_checkpoint_sha256)
    dtype_counts = Counter(array.dtype.name for array in loaded.weights.values())
    prefix_counts = Counter(name.split(".", 1)[0] for name in loaded.weights)
    names_payload = "".join(f"{name}\n" for name in sorted(loaded.weights)).encode("utf-8")
    return {
        "schema_version": 1,
        "artifact": "gacweights",
        "path": str(source),
        "size_bytes": source.stat().st_size,
        "sha256": _file_sha256(source),
        "source_checkpoint_sha256": loaded.source_checkpoint_sha256,
        "weight_count": len(loaded.weights),
        "tensor_bytes": sum(array.nbytes for array in loaded.weights.values()),
        "dtype_counts": dict(sorted(dtype_counts.items())),
        "top_level_prefix_counts": dict(sorted(prefix_counts.items())),
        "sorted_weight_names_sha256": sha256(names_payload).hexdigest(),
        "verified": True,
    }


def build_backbone_package(
    weights_path: str | Path,
    output_path: str | Path,
    *,
    expected_checkpoint_sha256: str | None = None,
    profile: ResNetFpnProfile = BEVFORMER_R50_FPN_PROFILE,
) -> dict[str, Any]:
    """Compile the fixed ResNet/FPN stage and write a verified HXP package."""

    source = Path(weights_path)
    destination = Path(output_path)
    if destination.suffix.lower() != ".hxp":
        raise ValidationError("BEVFormer backbone output must end in .hxp")
    loaded = load_weights(source)
    _check_provenance(loaded, expected_checkpoint_sha256)
    graph = build_bevformer_backbone_graph(loaded.weights, profile)
    program = compile_cuda(graph)
    model = (
        "BEVFormer-tiny"
        if profile == BEVFORMER_R50_FPN_PROFILE
        else "BEVFormer-custom-static"
    )
    metadata = {
        "model": model,
        "component": "img_backbone+img_neck",
        "profile": profile.to_metadata(),
        "source_weights_sha256": _file_sha256(source),
        "source_checkpoint_sha256": loaded.source_checkpoint_sha256,
        "validation_state": "offline-codegen-and-package-roundtrip",
    }
    write_package(destination, program, metadata=metadata)

    # A package is not considered built until the strict reader accepts it and
    # deterministic code generation reproduces its source and launch metadata.
    verified = load_package(destination)
    rebuilt = rebuild_cuda_program(verified)
    op_counts = Counter(node.op for node in verified.graph.nodes)
    return {
        "schema_version": 1,
        "artifact": "hxp",
        "model": model,
        "component": "img_backbone+img_neck",
        "profile": profile.name,
        "output": str(destination),
        "size_bytes": destination.stat().st_size,
        "sha256": _file_sha256(destination),
        "source_checkpoint_sha256": loaded.source_checkpoint_sha256,
        "node_count": len(verified.graph.nodes),
        "op_counts": dict(sorted(op_counts.items())),
        "input": verified.graph.inputs["images"].to_dict(),
        "output_spec": verified.graph.value_specs[verified.graph.outputs[0]].to_dict(),
        "constant_count": len(verified.graph.constants),
        "constant_bytes": sum(value.nbytes for value in verified.graph.constants.values()),
        "arena_bytes": verified.memory_plan.arena_size,
        "launch_count": len(rebuilt.launches),
        "codegen_verified": True,
        "physical_cuda_execution": False,
    }


def build_feature_prep_package(
    weights_path: str | Path,
    output_path: str | Path,
    *,
    expected_checkpoint_sha256: str | None = None,
    config: BevFormerFeaturePrepConfig | None = None,
) -> dict[str, Any]:
    """Compile FPN layout conversion and learned camera/level embeddings."""

    source, destination = Path(weights_path), Path(output_path)
    loaded = load_weights(source)
    _check_provenance(loaded, expected_checkpoint_sha256)
    selected_config = config or BevFormerFeaturePrepConfig()
    canonical = extract_feature_prep_weights(loaded.weights, selected_config)
    graph = build_bevformer_feature_prep_graph(
        canonical,
        selected_config,
        name="bevformer_single_level_feature_preparation",
    )
    return _write_component_package(
        loaded=loaded,
        source=source,
        destination=destination,
        graph=graph,
        component="encoder.feature_preparation",
        layer=None,
        model=(
            "BEVFormer-tiny"
            if selected_config == BevFormerFeaturePrepConfig()
            else "BEVFormer-custom-static"
        ),
    )


def build_encoder_inputs_package(
    weights_path: str | Path,
    output_path: str | Path,
    *,
    expected_checkpoint_sha256: str | None = None,
    config: BevFormerEncoderInputConfig | None = None,
) -> dict[str, Any]:
    """Compile learned BEV queries, CAN-bus conditioning, and static grids."""

    source, destination = Path(weights_path), Path(output_path)
    loaded = load_weights(source)
    _check_provenance(loaded, expected_checkpoint_sha256)
    selected_config = config or BevFormerEncoderInputConfig()
    canonical = extract_encoder_input_weights(loaded.weights, selected_config)
    graph = build_bevformer_encoder_input_graph(
        canonical,
        selected_config,
        name="bevformer_encoder_input_preparation",
    )
    return _write_component_package(
        loaded=loaded,
        source=source,
        destination=destination,
        graph=graph,
        component="encoder.input_preparation",
        layer=None,
        model=(
            "BEVFormer-tiny"
            if selected_config == BevFormerEncoderInputConfig()
            else "BEVFormer-custom-static"
        ),
    )


def _write_component_package(
    *,
    loaded: LoadedWeights,
    source: Path,
    destination: Path,
    graph: Any,
    component: str,
    layer: int | None,
    model: str,
    component_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if destination.suffix.lower() != ".hxp":
        raise ValidationError("BEVFormer component output must end in .hxp")
    stack = component.split(".", 1)[0]
    program = compile_cuda(graph)
    metadata: dict[str, Any] = {
        "model": model,
        "component": component,
        "stack": stack,
        "source_weights_sha256": _file_sha256(source),
        "source_checkpoint_sha256": loaded.source_checkpoint_sha256,
        "validation_state": "offline-codegen-and-package-roundtrip",
    }
    if layer is not None:
        metadata["layer_index"] = layer
    if component_metadata:
        metadata.update(component_metadata)
    write_package(destination, program, metadata=metadata)
    verified = load_package(destination)
    rebuilt = rebuild_cuda_program(verified)
    op_counts = Counter(node.op for node in verified.graph.nodes)
    report = {
        "schema_version": 1,
        "artifact": "hxp",
        "model": model,
        "component": component,
        "stack": stack,
        "output": str(destination),
        "size_bytes": destination.stat().st_size,
        "sha256": _file_sha256(destination),
        "source_checkpoint_sha256": loaded.source_checkpoint_sha256,
        "node_count": len(verified.graph.nodes),
        "op_counts": dict(sorted(op_counts.items())),
        "inputs": {
            name: spec.to_dict() for name, spec in verified.graph.inputs.items()
        },
        "outputs": {
            name: verified.graph.value_specs[name].to_dict()
            for name in verified.graph.outputs
        },
        "constant_count": len(verified.graph.constants),
        "constant_bytes": sum(value.nbytes for value in verified.graph.constants.values()),
        "arena_bytes": verified.memory_plan.arena_size,
        "launch_count": len(rebuilt.launches),
        "codegen_verified": True,
        "physical_cuda_execution": False,
    }
    if layer is not None:
        report["layer_index"] = layer
    if component_metadata:
        report.update(component_metadata)
    return report


def build_temporal_attention_package(
    weights_path: str | Path,
    output_path: str | Path,
    *,
    layer: int,
    expected_checkpoint_sha256: str | None = None,
    config: TemporalSelfAttentionConfig | None = None,
) -> dict[str, Any]:
    """Compile one official encoder TemporalSelfAttention parameterization."""

    source, destination = Path(weights_path), Path(output_path)
    loaded = load_weights(source)
    _check_provenance(loaded, expected_checkpoint_sha256)
    selected_config = config or TemporalSelfAttentionConfig()
    canonical = extract_temporal_attention_weights(
        loaded.weights, layer, selected_config
    )
    graph = build_temporal_self_attention_graph(
        canonical,
        selected_config,
        name=f"bevformer_encoder_layer_{layer}_temporal_attention",
    )
    return _write_component_package(
        loaded=loaded,
        source=source,
        destination=destination,
        graph=graph,
        component="encoder.temporal_self_attention",
        layer=layer,
        model=(
            "BEVFormer-tiny"
            if selected_config == TemporalSelfAttentionConfig()
            else "BEVFormer-custom-static"
        ),
    )


def build_spatial_attention_package(
    weights_path: str | Path,
    output_path: str | Path,
    *,
    layer: int,
    expected_checkpoint_sha256: str | None = None,
    config: SpatialCrossAttentionConfig | None = None,
) -> dict[str, Any]:
    """Compile one fixed-upper-bound SpatialCrossAttention parameterization."""

    source, destination = Path(weights_path), Path(output_path)
    loaded = load_weights(source)
    _check_provenance(loaded, expected_checkpoint_sha256)
    selected_config = config or SpatialCrossAttentionConfig()
    canonical = extract_spatial_attention_weights(
        loaded.weights, layer, selected_config
    )
    graph = build_spatial_cross_attention_graph(
        canonical,
        selected_config,
        name=f"bevformer_encoder_layer_{layer}_spatial_attention",
    )
    return _write_component_package(
        loaded=loaded,
        source=source,
        destination=destination,
        graph=graph,
        component="encoder.spatial_cross_attention",
        layer=layer,
        model=(
            "BEVFormer-tiny"
            if selected_config == SpatialCrossAttentionConfig()
            else "BEVFormer-custom-static"
        ),
    )


def build_decoder_cross_attention_package(
    weights_path: str | Path,
    output_path: str | Path,
    *,
    layer: int,
    expected_checkpoint_sha256: str | None = None,
    config: DecoderCrossAttentionConfig | None = None,
) -> dict[str, Any]:
    """Compile one decoder CustomMSDeformableAttention invocation."""

    source, destination = Path(weights_path), Path(output_path)
    loaded = load_weights(source)
    _check_provenance(loaded, expected_checkpoint_sha256)
    selected_config = config or DecoderCrossAttentionConfig()
    canonical = extract_decoder_cross_attention_weights(
        loaded.weights, layer, selected_config
    )
    graph = build_decoder_cross_attention_graph(
        canonical,
        selected_config,
        name=f"bevformer_decoder_layer_{layer}_cross_attention",
    )
    return _write_component_package(
        loaded=loaded,
        source=source,
        destination=destination,
        graph=graph,
        component="decoder.custom_msda_cross_attention",
        layer=layer,
        model=(
            "BEVFormer-tiny"
            if selected_config == DecoderCrossAttentionConfig()
            else "BEVFormer-custom-static"
        ),
    )


def build_encoder_ffn_package(
    weights_path: str | Path,
    output_path: str | Path,
    *,
    layer: int,
    expected_checkpoint_sha256: str | None = None,
    config: BEVFormerFFNConfig | None = None,
) -> dict[str, Any]:
    """Compile one encoder FFN together with its following LayerNorm."""

    source, destination = Path(weights_path), Path(output_path)
    loaded = load_weights(source)
    _check_provenance(loaded, expected_checkpoint_sha256)
    selected_config = config or BEVFormerFFNConfig()
    canonical = extract_encoder_ffn_weights(loaded.weights, layer, selected_config)
    graph = build_bevformer_ffn_graph(
        canonical,
        selected_config,
        name=f"bevformer_encoder_layer_{layer}_ffn",
    )
    return _write_component_package(
        loaded=loaded,
        source=source,
        destination=destination,
        graph=graph,
        component="encoder.ffn_and_norm2",
        layer=layer,
        model=(
            "BEVFormer-tiny"
            if selected_config == BEVFormerFFNConfig()
            else "BEVFormer-custom-static"
        ),
    )


def build_encoder_layer_package(
    weights_path: str | Path,
    output_path: str | Path,
    *,
    layer: int,
    expected_checkpoint_sha256: str | None = None,
    config: BevFormerEncoderLayerConfig | None = None,
) -> dict[str, Any]:
    """Compile one complete post-normalized BEVFormer encoder layer."""

    source, destination = Path(weights_path), Path(output_path)
    loaded = load_weights(source)
    _check_provenance(loaded, expected_checkpoint_sha256)
    selected_config = config or BevFormerEncoderLayerConfig()
    groups = extract_encoder_layer_weights(loaded.weights, layer, selected_config)
    graph = build_bevformer_encoder_layer_graph(
        groups["temporal"],
        groups["spatial"],
        groups["layer"],
        selected_config,
        name=f"bevformer_encoder_layer_{layer}",
    )
    return _write_component_package(
        loaded=loaded,
        source=source,
        destination=destination,
        graph=graph,
        component="encoder.complete_layer",
        layer=layer,
        model=(
            "BEVFormer-tiny"
            if selected_config == BevFormerEncoderLayerConfig()
            else "BEVFormer-custom-static"
        ),
    )


def build_encoder_stack_package(
    weights_path: str | Path,
    output_path: str | Path,
    *,
    expected_checkpoint_sha256: str | None = None,
    config: BevFormerEncoderStackConfig | None = None,
) -> dict[str, Any]:
    """Compile the complete fixed three-layer BEVFormer encoder stack."""

    source, destination = Path(weights_path), Path(output_path)
    loaded = load_weights(source)
    _check_provenance(loaded, expected_checkpoint_sha256)
    selected_config = config or BevFormerEncoderStackConfig()
    groups = [
        extract_encoder_layer_weights(
            loaded.weights,
            layer,
            selected_config.layer,
        )
        for layer in range(selected_config.num_layers)
    ]
    graph = build_bevformer_encoder_stack_graph(
        [group["temporal"] for group in groups],
        [group["spatial"] for group in groups],
        [group["layer"] for group in groups],
        selected_config,
        name="bevformer_encoder_stack",
    )
    return _write_component_package(
        loaded=loaded,
        source=source,
        destination=destination,
        graph=graph,
        component="encoder.complete_stack",
        layer=None,
        model=(
            "BEVFormer-tiny"
            if selected_config == BevFormerEncoderStackConfig()
            else "BEVFormer-custom-static"
        ),
    )


def _extract_history_pipeline_weight_groups(
    loaded: LoadedWeights,
    config: BevFormerHistoryPipelineConfig,
) -> dict[str, Any]:
    """Assemble a warm-history encoder only from verified data tensors.

    ``LoadedWeights`` is produced by the bounded, data-only ``.gacweights``
    reader.  Keeping this assembly function on that type makes the package
    builder's trust boundary explicit and avoids framework or pickle loading.
    """

    if not isinstance(loaded, LoadedWeights):
        raise ValidationError("history encoder weights must be verified LoadedWeights")
    if not isinstance(config, BevFormerHistoryPipelineConfig):
        raise ValidationError(
            "history encoder config must be a BevFormerHistoryPipelineConfig"
        )
    stack = config.encoder_stack
    layers = [
        extract_encoder_layer_weights(loaded.weights, index, stack.layer)
        for index in range(stack.num_layers)
    ]
    return {
        "backbone": validate_checkpoint_weights(loaded.weights, config.backbone),
        "feature_prep": extract_feature_prep_weights(
            loaded.weights,
            config.feature_prep,
        ),
        "encoder_inputs": extract_encoder_input_weights(
            loaded.weights,
            config.encoder_inputs,
        ),
        "temporal": tuple(group["temporal"] for group in layers),
        "spatial": tuple(group["spatial"] for group in layers),
        "layer": tuple(group["layer"] for group in layers),
    }


def build_history_encoder_package(
    weights_path: str | Path,
    output_path: str | Path,
    *,
    expected_checkpoint_sha256: str | None = None,
    config: BevFormerHistoryPipelineConfig | None = None,
) -> dict[str, Any]:
    """Compile the image-to-BEV encoder after external motion compensation."""

    source, destination = Path(weights_path), Path(output_path)
    loaded = load_weights(source)
    _check_provenance(loaded, expected_checkpoint_sha256)
    selected_config = config or BevFormerHistoryPipelineConfig()
    groups = _extract_history_pipeline_weight_groups(loaded, selected_config)
    graph = build_bevformer_history_pipeline_graph(
        groups["backbone"],
        groups["feature_prep"],
        groups["encoder_inputs"],
        groups["temporal"],
        groups["spatial"],
        groups["layer"],
        selected_config,
        name="bevformer_history_image_to_bev_encoder",
    )
    return _write_component_package(
        loaded=loaded,
        source=source,
        destination=destination,
        graph=graph,
        component="history.image_to_bev_encoder",
        layer=None,
        model=(
            "BEVFormer-tiny-warm-history-encoder"
            if selected_config == BevFormerHistoryPipelineConfig()
            else "BEVFormer-custom-warm-history-static"
        ),
        component_metadata={
            "warm_history_encoder": True,
            "motion_compensation_boundary": "external-before-this-graph",
            "required_aligned_history_inputs": [
                "aligned_prev_bev",
                "prev_reference_points",
            ],
            "num_encoder_layers": selected_config.encoder_stack.num_layers,
        },
    )


def build_decoder_self_attention_package(
    weights_path: str | Path,
    output_path: str | Path,
    *,
    layer: int,
    expected_checkpoint_sha256: str | None = None,
    config: DecoderSelfAttentionConfig | None = None,
) -> dict[str, Any]:
    """Compile one decoder self-attention plus norms.0 fragment."""

    source, destination = Path(weights_path), Path(output_path)
    loaded = load_weights(source)
    _check_provenance(loaded, expected_checkpoint_sha256)
    selected_config = config or DecoderSelfAttentionConfig()
    canonical = extract_decoder_self_attention_weights(
        loaded.weights, layer, selected_config
    )
    graph = build_decoder_self_attention_graph(
        canonical,
        selected_config,
        name=f"bevformer_decoder_layer_{layer}_self_attention",
    )
    return _write_component_package(
        loaded=loaded,
        source=source,
        destination=destination,
        graph=graph,
        component="decoder.self_attention_and_norm0",
        layer=layer,
        model=(
            "BEVFormer-tiny"
            if selected_config == DecoderSelfAttentionConfig()
            else "BEVFormer-custom-static"
        ),
    )


def build_decoder_layer_package(
    weights_path: str | Path,
    output_path: str | Path,
    *,
    layer: int,
    expected_checkpoint_sha256: str | None = None,
    config: BevFormerDecoderLayerConfig | None = None,
) -> dict[str, Any]:
    """Compile one complete post-normalized decoder transformer layer."""

    source, destination = Path(weights_path), Path(output_path)
    loaded = load_weights(source)
    _check_provenance(loaded, expected_checkpoint_sha256)
    selected_config = config or BevFormerDecoderLayerConfig()
    groups = extract_decoder_layer_weights(
        loaded.weights,
        layer,
        selected_config,
    )
    graph = build_bevformer_decoder_layer_graph(
        groups["self_attention"],
        groups["cross_attention"],
        groups["layer"],
        selected_config,
        name=f"bevformer_decoder_layer_{layer}",
    )
    return _write_component_package(
        loaded=loaded,
        source=source,
        destination=destination,
        graph=graph,
        component="decoder.complete_layer",
        layer=layer,
        model=(
            "BEVFormer-tiny"
            if selected_config == BevFormerDecoderLayerConfig()
            else "BEVFormer-custom-static"
        ),
    )


def build_decoder_stack_package(
    weights_path: str | Path,
    output_path: str | Path,
    *,
    expected_checkpoint_sha256: str | None = None,
    config: BevFormerDecoderStackConfig | None = None,
) -> dict[str, Any]:
    """Compile learned query initialization and all six decoder layers."""

    source, destination = Path(weights_path), Path(output_path)
    loaded = load_weights(source)
    _check_provenance(loaded, expected_checkpoint_sha256)
    selected_config = config or BevFormerDecoderStackConfig()
    groups = extract_decoder_stack_weights(loaded.weights, selected_config)
    graph = build_bevformer_decoder_stack_graph(
        groups["query_init"],
        groups["self_attention"],
        groups["cross_attention"],
        groups["layer"],
        groups["refinement"],
        selected_config,
        name="bevformer_decoder_stack",
    )
    return _write_component_package(
        loaded=loaded,
        source=source,
        destination=destination,
        graph=graph,
        component="decoder.complete_stack",
        layer=None,
        model=(
            "BEVFormer-tiny"
            if selected_config == BevFormerDecoderStackConfig()
            else "BEVFormer-custom-static"
        ),
        component_metadata={
            "full_decoder_stack": True,
            "num_layers": selected_config.num_layers,
        },
    )


def build_decoder_refinement_package(
    weights_path: str | Path,
    output_path: str | Path,
    *,
    layer: int,
    expected_checkpoint_sha256: str | None = None,
    config: DecoderReferenceRefinementConfig | None = None,
) -> dict[str, Any]:
    """Compile one decoder regression branch and reference update."""

    source, destination = Path(weights_path), Path(output_path)
    loaded = load_weights(source)
    _check_provenance(loaded, expected_checkpoint_sha256)
    selected_config = config or DecoderReferenceRefinementConfig()
    canonical = extract_decoder_refinement_weights(
        loaded.weights, layer, selected_config
    )
    graph = build_decoder_reference_refinement_graph(
        canonical,
        selected_config,
        name=f"bevformer_decoder_layer_{layer}_reference_refinement",
    )
    return _write_component_package(
        loaded=loaded,
        source=source,
        destination=destination,
        graph=graph,
        component="decoder.regression_reference_refinement",
        layer=layer,
        model=(
            "BEVFormer-tiny"
            if selected_config == DecoderReferenceRefinementConfig()
            else "BEVFormer-custom-static"
        ),
    )


def build_decoder_query_init_package(
    weights_path: str | Path,
    output_path: str | Path,
    *,
    expected_checkpoint_sha256: str | None = None,
    config: DecoderQueryInitConfig | None = None,
) -> dict[str, Any]:
    """Compile learned decoder queries and initial reference points."""

    source, destination = Path(weights_path), Path(output_path)
    loaded = load_weights(source)
    _check_provenance(loaded, expected_checkpoint_sha256)
    selected_config = config or DecoderQueryInitConfig()
    canonical = extract_decoder_query_init_weights(loaded.weights, selected_config)
    graph = build_decoder_query_init_graph(canonical, selected_config)
    return _write_component_package(
        loaded=loaded,
        source=source,
        destination=destination,
        graph=graph,
        component="decoder.query_and_reference_initialization",
        layer=None,
        model=(
            "BEVFormer-tiny"
            if selected_config == DecoderQueryInitConfig()
            else "BEVFormer-custom-static"
        ),
    )


def build_detection_head_package(
    weights_path: str | Path,
    output_path: str | Path,
    *,
    layer: int,
    expected_checkpoint_sha256: str | None = None,
    config: DetectionHeadConfig | None = None,
) -> dict[str, Any]:
    """Compile one class branch and reference-aware raw box-code projection.

    The package emits per-query logits, scores, and ten-channel box codes. A
    later explicit NMS-free stage performs Top-K and size/yaw denormalization.
    """

    source, destination = Path(weights_path), Path(output_path)
    loaded = load_weights(source)
    _check_provenance(loaded, expected_checkpoint_sha256)
    selected_config = config or DetectionHeadConfig()
    canonical = extract_detection_head_weights(
        loaded.weights,
        layer,
        selected_config,
    )
    graph = build_detection_head_graph(
        canonical,
        selected_config,
        name=f"bevformer_detection_head_layer_{layer}",
    )
    return _write_component_package(
        loaded=loaded,
        source=source,
        destination=destination,
        graph=graph,
        component="decoder.raw_detection_head",
        layer=layer,
        model=(
            "BEVFormer-tiny"
            if selected_config == DetectionHeadConfig()
            else "BEVFormer-custom-static"
        ),
    )


def build_detection_stack_package(
    weights_path: str | Path,
    output_path: str | Path,
    *,
    expected_checkpoint_sha256: str | None = None,
    config: DetectionStackConfig | None = None,
) -> dict[str, Any]:
    """Compile all six raw detection projection branches.

    The package keeps the official NMS-free top-k selection outside the graph.
    """

    source, destination = Path(weights_path), Path(output_path)
    loaded = load_weights(source)
    _check_provenance(loaded, expected_checkpoint_sha256)
    selected_config = config or DetectionStackConfig()
    canonical = extract_detection_stack_weights(
        loaded.weights,
        selected_config,
    )
    graph = build_detection_stack_graph(
        canonical,
        selected_config,
        name="bevformer_detection_projection_stack",
    )
    return _write_component_package(
        loaded=loaded,
        source=source,
        destination=destination,
        graph=graph,
        component="decoder.detection_projection_stack",
        layer=None,
        model=(
            "BEVFormer-tiny"
            if selected_config == DetectionStackConfig()
            else "BEVFormer-custom-static"
        ),
        component_metadata={
            "full_detection_projection_stack": True,
            "num_layers": selected_config.num_layers,
            "nms_free_topk": "outside-this-graph",
        },
    )


def build_decoder_pipeline_package(
    weights_path: str | Path,
    output_path: str | Path,
    *,
    expected_checkpoint_sha256: str | None = None,
    config: BevFormerDecoderPipelineConfig | None = None,
) -> dict[str, Any]:
    """Compile the six-layer decoder directly connected to all raw heads."""

    source, destination = Path(weights_path), Path(output_path)
    loaded = load_weights(source)
    _check_provenance(loaded, expected_checkpoint_sha256)
    selected_config = config or BevFormerDecoderPipelineConfig()
    decoder = extract_decoder_stack_weights(
        loaded.weights,
        selected_config.decoder,
    )
    detection = extract_detection_stack_weights(
        loaded.weights,
        selected_config.detection,
    )
    graph = build_bevformer_decoder_pipeline_graph(
        decoder["query_init"],
        decoder["self_attention"],
        decoder["cross_attention"],
        decoder["layer"],
        decoder["refinement"],
        detection,
        selected_config,
        name="bevformer_decoder_raw_detection_pipeline",
    )
    return _write_component_package(
        loaded=loaded,
        source=source,
        destination=destination,
        graph=graph,
        component="decoder.complete_raw_detection_pipeline",
        layer=None,
        model=(
            "BEVFormer-tiny"
            if selected_config == BevFormerDecoderPipelineConfig()
            else "BEVFormer-custom-static"
        ),
        component_metadata={
            "full_decoder_pipeline": True,
            "num_decoder_layers": selected_config.decoder.num_layers,
            "num_detection_layers": selected_config.detection.num_layers,
            "host_postprocess": "decode_nms_free",
        },
    )


def build_cold_pipeline_package(
    weights_path: str | Path,
    output_path: str | Path,
    *,
    expected_checkpoint_sha256: str | None = None,
    config: BevFormerColdPipelineConfig | None = None,
) -> dict[str, Any]:
    """Compile the cold single-frame image-to-raw-detections core."""

    source, destination = Path(weights_path), Path(output_path)
    loaded = load_weights(source)
    _check_provenance(loaded, expected_checkpoint_sha256)
    selected_config = config or BevFormerColdPipelineConfig()
    encoder_config = selected_config.encoder
    encoder_groups = [
        extract_encoder_layer_weights(
            loaded.weights,
            index,
            encoder_config.encoder_stack.layer,
        )
        for index in range(encoder_config.encoder_stack.num_layers)
    ]
    encoder = {
        "backbone": validate_checkpoint_weights(
            loaded.weights,
            encoder_config.backbone,
        ),
        "feature_prep": extract_feature_prep_weights(
            loaded.weights,
            encoder_config.feature_prep,
        ),
        "encoder_inputs": extract_encoder_input_weights(
            loaded.weights,
            encoder_config.encoder_inputs,
        ),
        "temporal": tuple(group["temporal"] for group in encoder_groups),
        "spatial": tuple(group["spatial"] for group in encoder_groups),
        "layer": tuple(group["layer"] for group in encoder_groups),
    }
    decoder_stack = extract_decoder_stack_weights(
        loaded.weights,
        selected_config.decoder.decoder,
    )
    decoder = {
        **decoder_stack,
        "detection": extract_detection_stack_weights(
            loaded.weights,
            selected_config.decoder.detection,
        ),
    }
    graph = build_bevformer_cold_pipeline_graph(
        encoder,
        decoder,
        selected_config,
        name="bevformer_cold_image_to_raw_detections",
    )
    default_config = BevFormerColdPipelineConfig()
    return _write_component_package(
        loaded=loaded,
        source=source,
        destination=destination,
        graph=graph,
        component="cold.single_frame_image_to_raw_detections",
        layer=None,
        model=(
            "BEVFormer-tiny-cold-single-frame"
            if selected_config == default_config
            else "BEVFormer-custom-cold-static"
        ),
        component_metadata={
            "cold_single_frame_only": True,
            "streaming_step": False,
            "host_postprocess": "decode_nms_free",
            "num_encoder_layers": selected_config.encoder.encoder_stack.num_layers,
            "num_decoder_layers": selected_config.decoder.decoder.num_layers,
        },
    )


def _extract_step_pipeline_weight_groups(
    loaded: LoadedWeights,
    config: BevFormerStepPipelineConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Assemble the complete step solely from verified data tensors.

    The input is deliberately a ``LoadedWeights`` instance returned by the
    bounded ``.gacweights`` reader.  No framework checkpoint or pickle loader
    participates in this build path.
    """

    if not isinstance(loaded, LoadedWeights):
        raise ValidationError("streaming step weights must be verified LoadedWeights")
    if not isinstance(config, BevFormerStepPipelineConfig):
        raise ValidationError(
            "streaming step config must be a BevFormerStepPipelineConfig"
        )
    encoder = _extract_history_pipeline_weight_groups(loaded, config.encoder)
    decoder_stack = extract_decoder_stack_weights(
        loaded.weights,
        config.decoder.decoder,
    )
    decoder = {
        **decoder_stack,
        "detection": extract_detection_stack_weights(
            loaded.weights,
            config.decoder.detection,
        ),
    }
    return encoder, decoder


def build_streaming_step_package(
    weights_path: str | Path,
    output_path: str | Path,
    *,
    expected_checkpoint_sha256: str | None = None,
    config: BevFormerStepPipelineConfig | None = None,
) -> dict[str, Any]:
    """Compile the complete cold-or-warm BEVFormer streaming step."""

    source, destination = Path(weights_path), Path(output_path)
    loaded = load_weights(source)
    _check_provenance(loaded, expected_checkpoint_sha256)
    selected_config = config or BevFormerStepPipelineConfig()
    encoder, decoder = _extract_step_pipeline_weight_groups(
        loaded,
        selected_config,
    )
    graph = build_bevformer_step_pipeline_graph(
        encoder,
        decoder,
        selected_config,
        name="bevformer_complete_cold_and_warm_streaming_step",
    )
    default_config = BevFormerStepPipelineConfig()
    raw_box_representation = (
        "[physical-x,physical-y,log-w,log-l,physical-z,log-h,"
        "sin-yaw,cos-yaw,vx,vy]"
    )
    return _write_component_package(
        loaded=loaded,
        source=source,
        destination=destination,
        graph=graph,
        component="streaming.complete_cold_and_warm_step",
        layer=None,
        model=(
            "BEVFormer-tiny-streaming-step"
            if selected_config == default_config
            else "BEVFormer-custom-streaming-step-static"
        ),
        component_metadata={
            "streaming_step": True,
            "cold_and_warm": True,
            "host_postprocess": "decode_nms_free",
            "topk_boundary": "host-postprocess",
            "raw_box_representation": raw_box_representation,
            "num_encoder_layers": (
                selected_config.encoder.encoder_stack.num_layers
            ),
            "num_decoder_layers": selected_config.decoder.decoder.num_layers,
            "num_detection_layers": selected_config.decoder.detection.num_layers,
        },
    )


def build_contract_file(
    config_path: str | Path,
    output_path: str | Path,
    *,
    source_label: str | None = None,
) -> dict[str, Any]:
    """Audit the static tiny config and persist its pure-tensor step ABI."""

    source = Path(config_path)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    manifest = build_bevformer_tiny_manifest(
        source,
        source_label=source_label or source.as_posix(),
    )
    save_bevformer_manifest(manifest, destination)
    return {
        "schema_version": 1,
        "artifact": "bevformer-step-manifest",
        "output": str(destination),
        "sha256": _file_sha256(destination),
        "profile": manifest.profile,
        "config_sha256": manifest.config_sha256,
        "status": manifest.status,
        "input_count": len(manifest.inputs),
        "output_count": len(manifest.outputs),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build auditable BEVFormer-tiny artifacts without importing PyTorch."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    contract = commands.add_parser("contract", help="audit the tiny config and write its tensor ABI")
    contract.add_argument("--config", required=True, type=Path)
    contract.add_argument("-o", "--output", required=True, type=Path)
    contract.add_argument("--source-label")

    weights = commands.add_parser("inspect-weights", help="verify and inventory a .gacweights archive")
    weights.add_argument("weights", type=Path)
    weights.add_argument("--expected-checkpoint-sha256")

    backbone = commands.add_parser("backbone", help="compile the fixed R50 + one-level FPN package")
    backbone.add_argument("--weights", required=True, type=Path)
    backbone.add_argument("-o", "--output", required=True, type=Path)
    backbone.add_argument("--expected-checkpoint-sha256")

    feature_prep = commands.add_parser(
        "feature-prep",
        help="compile FPN layout conversion plus camera and level embeddings",
    )
    feature_prep.add_argument("--weights", required=True, type=Path)
    feature_prep.add_argument("-o", "--output", required=True, type=Path)
    feature_prep.add_argument("--expected-checkpoint-sha256")

    encoder_inputs = commands.add_parser(
        "encoder-inputs",
        help="compile BEV queries, CAN-bus conditioning, and static grids",
    )
    encoder_inputs.add_argument("--weights", required=True, type=Path)
    encoder_inputs.add_argument("-o", "--output", required=True, type=Path)
    encoder_inputs.add_argument("--expected-checkpoint-sha256")

    temporal = commands.add_parser(
        "temporal-attention",
        help="compile one encoder TemporalSelfAttention package",
    )
    temporal.add_argument("--weights", required=True, type=Path)
    temporal.add_argument("--layer", required=True, type=int, choices=range(3))
    temporal.add_argument("-o", "--output", required=True, type=Path)
    temporal.add_argument("--expected-checkpoint-sha256")

    spatial = commands.add_parser(
        "spatial-attention",
        help="compile one encoder fixed-upper-bound SpatialCrossAttention package",
    )
    spatial.add_argument("--weights", required=True, type=Path)
    spatial.add_argument("--layer", required=True, type=int, choices=range(3))
    spatial.add_argument("-o", "--output", required=True, type=Path)
    spatial.add_argument("--expected-checkpoint-sha256")

    ffn = commands.add_parser(
        "encoder-ffn",
        help="compile one encoder FFN plus its following LayerNorm package",
    )
    ffn.add_argument("--weights", required=True, type=Path)
    ffn.add_argument("--layer", required=True, type=int, choices=range(3))
    ffn.add_argument("-o", "--output", required=True, type=Path)
    ffn.add_argument("--expected-checkpoint-sha256")

    encoder_layer = commands.add_parser(
        "encoder-layer",
        help="compile one complete encoder layer including projection geometry",
    )
    encoder_layer.add_argument("--weights", required=True, type=Path)
    encoder_layer.add_argument("--layer", required=True, type=int, choices=range(3))
    encoder_layer.add_argument("-o", "--output", required=True, type=Path)
    encoder_layer.add_argument("--expected-checkpoint-sha256")

    encoder_stack = commands.add_parser(
        "encoder-stack",
        help="compile the complete three-layer encoder stack",
    )
    encoder_stack.add_argument("--weights", required=True, type=Path)
    encoder_stack.add_argument(
        "--mode",
        choices=("history", "cold"),
        default="history",
        help="use an explicit history queue or per-layer cold-start queues",
    )
    encoder_stack.add_argument("-o", "--output", required=True, type=Path)
    encoder_stack.add_argument("--expected-checkpoint-sha256")

    history_encoder = commands.add_parser(
        "history-encoder",
        help=(
            "compile the warm-history image-to-BEV encoder after external "
            "motion compensation"
        ),
    )
    history_encoder.add_argument("--weights", required=True, type=Path)
    history_encoder.add_argument("-o", "--output", required=True, type=Path)
    history_encoder.add_argument("--expected-checkpoint-sha256")

    decoder = commands.add_parser(
        "decoder-cross-attention",
        help="compile one decoder CustomMSDeformableAttention package",
    )
    decoder.add_argument("--weights", required=True, type=Path)
    decoder.add_argument("--layer", required=True, type=int, choices=range(6))
    decoder.add_argument("-o", "--output", required=True, type=Path)
    decoder.add_argument("--expected-checkpoint-sha256")

    decoder_self = commands.add_parser(
        "decoder-self-attention",
        help="compile one decoder self-attention plus norms.0 package",
    )
    decoder_self.add_argument("--weights", required=True, type=Path)
    decoder_self.add_argument("--layer", required=True, type=int, choices=range(6))
    decoder_self.add_argument("-o", "--output", required=True, type=Path)
    decoder_self.add_argument("--expected-checkpoint-sha256")

    decoder_layer = commands.add_parser(
        "decoder-layer",
        help="compile one complete decoder transformer layer",
    )
    decoder_layer.add_argument("--weights", required=True, type=Path)
    decoder_layer.add_argument("--layer", required=True, type=int, choices=range(6))
    decoder_layer.add_argument("-o", "--output", required=True, type=Path)
    decoder_layer.add_argument("--expected-checkpoint-sha256")

    decoder_stack = commands.add_parser(
        "decoder-stack",
        help="compile learned query initialization and all six decoder layers",
    )
    decoder_stack.add_argument("--weights", required=True, type=Path)
    decoder_stack.add_argument("-o", "--output", required=True, type=Path)
    decoder_stack.add_argument("--expected-checkpoint-sha256")

    refinement = commands.add_parser(
        "decoder-refinement",
        help="compile one decoder regression/reference-refinement package",
    )
    refinement.add_argument("--weights", required=True, type=Path)
    refinement.add_argument("--layer", required=True, type=int, choices=range(6))
    refinement.add_argument("-o", "--output", required=True, type=Path)
    refinement.add_argument("--expected-checkpoint-sha256")

    query_init = commands.add_parser(
        "decoder-query-init",
        help="compile learned decoder queries and initial references",
    )
    query_init.add_argument("--weights", required=True, type=Path)
    query_init.add_argument("-o", "--output", required=True, type=Path)
    query_init.add_argument("--expected-checkpoint-sha256")

    detection = commands.add_parser(
        "detection-head",
        help="compile one raw per-query classification and box decode head",
    )
    detection.add_argument("--weights", required=True, type=Path)
    detection.add_argument("--layer", required=True, type=int, choices=range(6))
    detection.add_argument("-o", "--output", required=True, type=Path)
    detection.add_argument("--expected-checkpoint-sha256")

    detection_stack = commands.add_parser(
        "detection-stack",
        help="compile all six raw detection projection branches",
    )
    detection_stack.add_argument("--weights", required=True, type=Path)
    detection_stack.add_argument("-o", "--output", required=True, type=Path)
    detection_stack.add_argument("--expected-checkpoint-sha256")

    decoder_pipeline = commands.add_parser(
        "decoder-pipeline",
        help="compile the six-layer decoder connected to all raw heads",
    )
    decoder_pipeline.add_argument("--weights", required=True, type=Path)
    decoder_pipeline.add_argument("-o", "--output", required=True, type=Path)
    decoder_pipeline.add_argument("--expected-checkpoint-sha256")

    cold_pipeline = commands.add_parser(
        "cold-pipeline",
        help="compile the cold single-frame image-to-raw-detections core",
    )
    cold_pipeline.add_argument("--weights", required=True, type=Path)
    cold_pipeline.add_argument("-o", "--output", required=True, type=Path)
    cold_pipeline.add_argument("--expected-checkpoint-sha256")

    step_pipeline = commands.add_parser(
        "step-pipeline",
        help="compile the complete cold-and-warm stateful streaming step",
    )
    step_pipeline.add_argument("--weights", required=True, type=Path)
    step_pipeline.add_argument("-o", "--output", required=True, type=Path)
    step_pipeline.add_argument("--expected-checkpoint-sha256")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    try:
        if arguments.command == "contract":
            report = build_contract_file(
                arguments.config,
                arguments.output,
                source_label=arguments.source_label,
            )
        elif arguments.command == "inspect-weights":
            report = inspect_bevformer_weights(
                arguments.weights,
                expected_checkpoint_sha256=arguments.expected_checkpoint_sha256,
            )
        elif arguments.command == "backbone":
            report = build_backbone_package(
                arguments.weights,
                arguments.output,
                expected_checkpoint_sha256=arguments.expected_checkpoint_sha256,
            )
        elif arguments.command == "feature-prep":
            report = build_feature_prep_package(
                arguments.weights,
                arguments.output,
                expected_checkpoint_sha256=arguments.expected_checkpoint_sha256,
            )
        elif arguments.command == "encoder-inputs":
            report = build_encoder_inputs_package(
                arguments.weights,
                arguments.output,
                expected_checkpoint_sha256=arguments.expected_checkpoint_sha256,
            )
        elif arguments.command == "temporal-attention":
            report = build_temporal_attention_package(
                arguments.weights,
                arguments.output,
                layer=arguments.layer,
                expected_checkpoint_sha256=arguments.expected_checkpoint_sha256,
            )
        elif arguments.command == "spatial-attention":
            report = build_spatial_attention_package(
                arguments.weights,
                arguments.output,
                layer=arguments.layer,
                expected_checkpoint_sha256=arguments.expected_checkpoint_sha256,
            )
        elif arguments.command == "encoder-ffn":
            report = build_encoder_ffn_package(
                arguments.weights,
                arguments.output,
                layer=arguments.layer,
                expected_checkpoint_sha256=arguments.expected_checkpoint_sha256,
            )
        elif arguments.command == "encoder-layer":
            report = build_encoder_layer_package(
                arguments.weights,
                arguments.output,
                layer=arguments.layer,
                expected_checkpoint_sha256=arguments.expected_checkpoint_sha256,
            )
        elif arguments.command == "encoder-stack":
            report = build_encoder_stack_package(
                arguments.weights,
                arguments.output,
                expected_checkpoint_sha256=arguments.expected_checkpoint_sha256,
                config=BevFormerEncoderStackConfig(
                    history_mode=arguments.mode == "history"
                ),
            )
        elif arguments.command == "history-encoder":
            report = build_history_encoder_package(
                arguments.weights,
                arguments.output,
                expected_checkpoint_sha256=arguments.expected_checkpoint_sha256,
            )
        elif arguments.command == "decoder-cross-attention":
            report = build_decoder_cross_attention_package(
                arguments.weights,
                arguments.output,
                layer=arguments.layer,
                expected_checkpoint_sha256=arguments.expected_checkpoint_sha256,
            )
        elif arguments.command == "decoder-self-attention":
            report = build_decoder_self_attention_package(
                arguments.weights,
                arguments.output,
                layer=arguments.layer,
                expected_checkpoint_sha256=arguments.expected_checkpoint_sha256,
            )
        elif arguments.command == "decoder-layer":
            report = build_decoder_layer_package(
                arguments.weights,
                arguments.output,
                layer=arguments.layer,
                expected_checkpoint_sha256=arguments.expected_checkpoint_sha256,
            )
        elif arguments.command == "decoder-stack":
            report = build_decoder_stack_package(
                arguments.weights,
                arguments.output,
                expected_checkpoint_sha256=arguments.expected_checkpoint_sha256,
            )
        elif arguments.command == "decoder-refinement":
            report = build_decoder_refinement_package(
                arguments.weights,
                arguments.output,
                layer=arguments.layer,
                expected_checkpoint_sha256=arguments.expected_checkpoint_sha256,
            )
        elif arguments.command == "decoder-query-init":
            report = build_decoder_query_init_package(
                arguments.weights,
                arguments.output,
                expected_checkpoint_sha256=arguments.expected_checkpoint_sha256,
            )
        elif arguments.command == "detection-head":
            report = build_detection_head_package(
                arguments.weights,
                arguments.output,
                layer=arguments.layer,
                expected_checkpoint_sha256=arguments.expected_checkpoint_sha256,
            )
        elif arguments.command == "detection-stack":
            report = build_detection_stack_package(
                arguments.weights,
                arguments.output,
                expected_checkpoint_sha256=arguments.expected_checkpoint_sha256,
            )
        elif arguments.command == "decoder-pipeline":
            report = build_decoder_pipeline_package(
                arguments.weights,
                arguments.output,
                expected_checkpoint_sha256=arguments.expected_checkpoint_sha256,
            )
        elif arguments.command == "cold-pipeline":
            report = build_cold_pipeline_package(
                arguments.weights,
                arguments.output,
                expected_checkpoint_sha256=arguments.expected_checkpoint_sha256,
            )
        else:
            report = build_streaming_step_package(
                arguments.weights,
                arguments.output,
                expected_checkpoint_sha256=arguments.expected_checkpoint_sha256,
            )
    except (GACError, ValidationError, BevFormerBackboneError, BevFormerExportError, OSError) as exc:
        print(f"gac-bevformer: {exc}", file=sys.stderr)
        return 1
    json.dump(report, sys.stdout, ensure_ascii=False, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
