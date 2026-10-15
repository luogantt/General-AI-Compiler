"""Import the complete BEVFormer streaming graph from data-only weights.

This frontend assembles backbone/FPN, all encoder and decoder layers, motion
alignment, state selection and raw detection heads. It returns the common Graph
IR without selecting a backend or creating a CUDA/HXP package. It never loads
a framework checkpoint, imports PyTorch/MMCV or executes pickle data.
"""

from __future__ import annotations

from hashlib import sha256
import hmac
from pathlib import Path
import re
from typing import Any

from ..bevformer_step_pipeline import BevFormerStepPipelineConfig, build_bevformer_step_pipeline_graph
from ..errors import ValidationError
from ..ir import Graph
from ..weights import LoadedWeights, load_weights
from .bevformer_backbone import validate_checkpoint_weights
from .bevformer_weights import (
    extract_decoder_stack_weights,
    extract_detection_stack_weights,
    extract_encoder_input_weights,
    extract_encoder_layer_weights,
    extract_feature_prep_weights,
)


def _weight_groups(loaded: LoadedWeights, config: BevFormerStepPipelineConfig) -> tuple[dict[str, Any], dict[str, Any]]:
    """Map checkpoint tensor names into the complete graph's learned groups."""
    if not isinstance(loaded, LoadedWeights):
        raise ValidationError("BEVFormer graph import requires verified LoadedWeights")
    encoder = config.encoder
    stack = encoder.encoder_stack
    layers = [extract_encoder_layer_weights(loaded.weights, index, stack.layer)
              for index in range(stack.num_layers)]
    encoder_groups = {
        "backbone": validate_checkpoint_weights(loaded.weights, encoder.backbone),
        "feature_prep": extract_feature_prep_weights(loaded.weights, encoder.feature_prep),
        "encoder_inputs": extract_encoder_input_weights(loaded.weights, encoder.encoder_inputs),
        "temporal": tuple(layer["temporal"] for layer in layers),
        "spatial": tuple(layer["spatial"] for layer in layers),
        "layer": tuple(layer["layer"] for layer in layers),
    }
    decoder_groups = {
        **extract_decoder_stack_weights(loaded.weights, config.decoder.decoder),
        "detection": extract_detection_stack_weights(loaded.weights, config.decoder.detection),
    }
    return encoder_groups, decoder_groups


def load_bevformer_step_graph(
    weights_path: str | Path,
    config: BevFormerStepPipelineConfig | None = None,
    expected_checkpoint_sha256: str | None = None,
) -> Graph:
    """Read a verified .gacweights archive and build the full static step IR.

    ``expected_checkpoint_sha256`` checks the source checkpoint provenance in
    the weight manifest, rather than the archive's own digest. Archive members
    and their tensor payloads are separately verified by ``load_weights``.
    The returned graph ends at raw detections and next_bev; NMS-free Top-K box
    decoding remains an explicit postprocessing boundary outside this graph.
    """
    if config is not None and not isinstance(config, BevFormerStepPipelineConfig):
        raise ValidationError("config must be a BevFormerStepPipelineConfig")
    selected = config if config is not None else BevFormerStepPipelineConfig()
    if expected_checkpoint_sha256 is not None and (
        not isinstance(expected_checkpoint_sha256, str)
        or re.fullmatch(r"[0-9a-fA-F]{64}", expected_checkpoint_sha256) is None
    ):
        raise ValidationError("expected checkpoint SHA256 must contain 64 hexadecimal characters")
    source = Path(weights_path)
    loaded = load_weights(source)
    if expected_checkpoint_sha256 is not None and not hmac.compare_digest(
        expected_checkpoint_sha256.lower(), loaded.source_checkpoint_sha256
    ):
        raise ValidationError("weights provenance mismatch: expected checkpoint "
                              f"{expected_checkpoint_sha256.lower()}, got {loaded.source_checkpoint_sha256}")
    encoder, decoder = _weight_groups(loaded, selected)
    graph = build_bevformer_step_pipeline_graph(
        encoder, decoder, selected, name="bevformer_complete_cold_and_warm_streaming_step",
    )
    archive_digest = sha256()
    with source.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            archive_digest.update(chunk)
    graph.metadata.update({
        "importer": "bevformer-step-data-only",
        "source_weights_format": "gacweights",
        "source_weights_path": str(source),
        "source_weights_sha256": archive_digest.hexdigest(),
        "source_checkpoint_sha256": loaded.source_checkpoint_sha256,
        "loaded_weight_count": len(loaded),
        "component": "streaming.complete_cold_and_warm_step",
        "model": ("BEVFormer-tiny-streaming-step" if config is None
                  else "BEVFormer-static-streaming-step"),
        "host_postprocess": "decode_nms_free",
        "postprocess": "explicit host NMS-free Top-K and box decoding after raw graph outputs",
    })
    graph.validate()
    return graph
