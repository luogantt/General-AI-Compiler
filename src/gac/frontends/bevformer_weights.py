"""Map official BEVFormer checkpoint names to canonical component weights."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from ..bevformer_ffn import BEVFormerFFNConfig, ffn_weight_shapes
from ..bevformer_spatial import SpatialCrossAttentionConfig
from ..bevformer_temporal import TemporalSelfAttentionConfig, temporal_weight_shapes
from ..bevformer_decoder import (
    DecoderCrossAttentionConfig,
    decoder_cross_attention_weight_shapes,
)
from ..bevformer_refinement import (
    DecoderReferenceRefinementConfig,
    decoder_refinement_weight_shapes,
)
from ..bevformer_decoder_self import DecoderSelfAttentionConfig
from ..bevformer_decoder_layer import BevFormerDecoderLayerConfig
from ..bevformer_detection_head import DetectionHeadConfig
from ..bevformer_encoder_layer import BevFormerEncoderLayerConfig
from ..bevformer_query_init import DecoderQueryInitConfig
from ..bevformer_feature_prep import (
    BevFormerFeaturePrepConfig,
    feature_prep_weight_shapes,
)
from ..bevformer_encoder_inputs import BevFormerEncoderInputConfig
from ..bevformer_decoder_stack import BevFormerDecoderStackConfig
from ..bevformer_detection_stack import DetectionStackConfig
from ..errors import ValidationError


ENCODER_PREFIX = "pts_bbox_head.transformer.encoder.layers"
DECODER_PREFIX = "pts_bbox_head.transformer.decoder.layers"


def _layer_index(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValidationError("BEVFormer layer must be a non-negative integer")
    return value


def _mapping(weights: Mapping[str, np.ndarray]) -> Mapping[str, np.ndarray]:
    if not isinstance(weights, Mapping):
        raise ValidationError("checkpoint weights must be a mapping")
    if any(not isinstance(name, str) or not name for name in weights):
        raise ValidationError("checkpoint weight names must be non-empty strings")
    return weights


def _tensor(
    weights: Mapping[str, np.ndarray],
    name: str,
    shape: tuple[int, ...],
) -> np.ndarray:
    if name not in weights:
        raise ValidationError(f"checkpoint is missing tensor {name!r}")
    value = weights[name]
    if not isinstance(value, np.ndarray):
        raise ValidationError(f"checkpoint tensor {name!r} must be a NumPy array")
    if value.shape != shape or value.dtype != np.float32:
        raise ValidationError(
            f"checkpoint tensor {name!r} requires {shape}/float32, "
            f"got {value.shape}/{value.dtype}"
        )
    if not value.flags.c_contiguous or not np.isfinite(value).all():
        raise ValidationError(f"checkpoint tensor {name!r} must be contiguous and finite")
    return value


def _leading_rows_tensor(
    weights: Mapping[str, np.ndarray],
    name: str,
    shape: tuple[int, ...],
    *,
    allowed_source_rows: tuple[int, ...],
) -> np.ndarray:
    """Read a checkpoint tensor whose unused trailing rows may be persisted.

    Some official modules allocate parameters for a fixed maximum number of
    feature levels while a concrete model feeds fewer levels.  The accepted
    source row counts remain explicit so a different model profile cannot be
    silently interpreted as the compiled target.  Rank, trailing dimensions,
    dtype, contiguity, and finiteness remain exact checkpoint requirements.
    """

    if name not in weights:
        raise ValidationError(f"checkpoint is missing tensor {name!r}")
    value = weights[name]
    if not isinstance(value, np.ndarray):
        raise ValidationError(f"checkpoint tensor {name!r} must be a NumPy array")
    if (
        value.ndim != len(shape)
        or value.shape[0] not in allowed_source_rows
        or value.shape[1:] != shape[1:]
        or value.dtype != np.float32
    ):
        row_counts = " or ".join(str(count) for count in allowed_source_rows)
        raise ValidationError(
            f"checkpoint tensor {name!r} requires leading row count "
            f"{row_counts} with trailing shape {shape[1:]}/float32, "
            f"got {value.shape}/{value.dtype}"
        )
    if not value.flags.c_contiguous or not np.isfinite(value).all():
        raise ValidationError(f"checkpoint tensor {name!r} must be contiguous and finite")
    return value[: shape[0]]


def _linear(
    weights: Mapping[str, np.ndarray],
    prefix: str,
    input_channels: int,
    output_channels: int,
) -> tuple[np.ndarray, np.ndarray]:
    # torch.nn.Linear persists [out, in]; canonical MatMul consumes [in, out].
    weight = _tensor(
        weights,
        f"{prefix}.weight",
        (output_channels, input_channels),
    )
    bias = _tensor(weights, f"{prefix}.bias", (output_channels,))
    return np.ascontiguousarray(weight.T), bias.copy()


def extract_temporal_attention_weights(
    weights: Mapping[str, np.ndarray],
    layer: int,
    config: TemporalSelfAttentionConfig | None = None,
    *,
    encoder_prefix: str = ENCODER_PREFIX,
) -> dict[str, np.ndarray]:
    """Select and transpose one official TemporalSelfAttention parameter set."""

    weights = _mapping(weights)
    layer = _layer_index(layer)
    config = config or TemporalSelfAttentionConfig()
    if not isinstance(config, TemporalSelfAttentionConfig):
        raise ValidationError("config must be a TemporalSelfAttentionConfig")
    if not isinstance(encoder_prefix, str) or not encoder_prefix:
        raise ValidationError("encoder_prefix must be non-empty")
    base = f"{encoder_prefix}.{layer}.attentions.0"
    shapes = temporal_weight_shapes(config)
    result: dict[str, np.ndarray] = {}
    for canonical, module in (
        ("value", "value_proj"),
        ("offset", "sampling_offsets"),
        ("attention", "attention_weights"),
        ("output", "output_proj"),
    ):
        input_channels, output_channels = shapes[f"{canonical}_weight"]
        matrix, bias = _linear(
            weights,
            f"{base}.{module}",
            input_channels,
            output_channels,
        )
        result[f"{canonical}_weight"] = matrix
        result[f"{canonical}_bias"] = bias
    return result


def _spatial_shapes(config: SpatialCrossAttentionConfig) -> dict[str, tuple[int, int]]:
    channels = config.embed_dim
    offset_channels = config.heads * config.levels * config.sampling_points * 2
    attention_channels = config.heads * config.levels * config.sampling_points
    return {
        "value": (channels, channels),
        "offset": (channels, offset_channels),
        "attention": (channels, attention_channels),
        "output": (channels, channels),
    }


def extract_spatial_attention_weights(
    weights: Mapping[str, np.ndarray],
    layer: int,
    config: SpatialCrossAttentionConfig | None = None,
    *,
    encoder_prefix: str = ENCODER_PREFIX,
) -> dict[str, np.ndarray]:
    """Select one SpatialCrossAttention plus its nested MSDA projections."""

    weights = _mapping(weights)
    layer = _layer_index(layer)
    config = config or SpatialCrossAttentionConfig()
    if not isinstance(config, SpatialCrossAttentionConfig):
        raise ValidationError("config must be a SpatialCrossAttentionConfig")
    if not isinstance(encoder_prefix, str) or not encoder_prefix:
        raise ValidationError("encoder_prefix must be non-empty")
    outer = f"{encoder_prefix}.{layer}.attentions.1"
    deformable = f"{outer}.deformable_attention"
    result: dict[str, np.ndarray] = {}
    for canonical, module, prefix in (
        ("value", "value_proj", deformable),
        ("offset", "sampling_offsets", deformable),
        ("attention", "attention_weights", deformable),
        ("output", "output_proj", outer),
    ):
        input_channels, output_channels = _spatial_shapes(config)[canonical]
        matrix, bias = _linear(
            weights,
            f"{prefix}.{module}",
            input_channels,
            output_channels,
        )
        result[f"{canonical}_weight"] = matrix
        result[f"{canonical}_bias"] = bias
    return result


def extract_encoder_ffn_weights(
    weights: Mapping[str, np.ndarray],
    layer: int,
    config: BEVFormerFFNConfig | None = None,
    *,
    encoder_prefix: str = ENCODER_PREFIX,
) -> dict[str, np.ndarray]:
    """Select one encoder FFN and its following post-normalization weights.

    Official PyTorch Linear matrices are transposed from ``[out, in]`` to the
    canonical ``[in, out]`` layout.  For BEVFormer's post-normalization
    operation order, ``norms.2`` is selected only when the static FFN graph
    includes its following ``norm`` operation.
    """

    weights = _mapping(weights)
    layer = _layer_index(layer)
    if config is None:
        config = BEVFormerFFNConfig()
    if not isinstance(config, BEVFormerFFNConfig):
        raise ValidationError("config must be a BEVFormerFFNConfig")
    if not isinstance(encoder_prefix, str) or not encoder_prefix:
        raise ValidationError("encoder_prefix must be non-empty")

    base = f"{encoder_prefix}.{layer}"
    shapes = ffn_weight_shapes(config)
    result: dict[str, np.ndarray] = {}
    for canonical, module in (
        ("fc1", "ffns.0.layers.0.0"),
        ("fc2", "ffns.0.layers.1"),
    ):
        input_channels, output_channels = shapes[f"{canonical}_weight"]
        matrix, bias = _linear(
            weights,
            f"{base}.{module}",
            input_channels,
            output_channels,
        )
        result[f"{canonical}_weight"] = matrix
        result[f"{canonical}_bias"] = bias

    if config.post_layer_norm:
        norm = f"{base}.norms.2"
        channels = config.embed_dim
        result["norm_weight"] = _tensor(
            weights, f"{norm}.weight", (channels,)
        ).copy()
        result["norm_bias"] = _tensor(
            weights, f"{norm}.bias", (channels,)
        ).copy()
    return result


def extract_encoder_layer_weights(
    weights: Mapping[str, np.ndarray],
    layer: int,
    config: BevFormerEncoderLayerConfig | None = None,
    *,
    encoder_prefix: str = ENCODER_PREFIX,
) -> dict[str, dict[str, np.ndarray]]:
    """Map every learned tensor used by one complete encoder layer graph."""

    weights = _mapping(weights)
    layer = _layer_index(layer)
    config = config or BevFormerEncoderLayerConfig()
    if not isinstance(config, BevFormerEncoderLayerConfig):
        raise ValidationError("config must be a BevFormerEncoderLayerConfig")
    if not isinstance(encoder_prefix, str) or not encoder_prefix:
        raise ValidationError("encoder_prefix must be non-empty")
    temporal = extract_temporal_attention_weights(
        weights,
        layer,
        config.temporal,
        encoder_prefix=encoder_prefix,
    )
    spatial = extract_spatial_attention_weights(
        weights,
        layer,
        config.spatial,
        encoder_prefix=encoder_prefix,
    )
    base = f"{encoder_prefix}.{layer}"
    channels = config.embed_dim
    hidden = config.ffn_hidden_dim
    fc1_weight, fc1_bias = _linear(
        weights, f"{base}.ffns.0.layers.0.0", channels, hidden
    )
    fc2_weight, fc2_bias = _linear(
        weights, f"{base}.ffns.0.layers.1", hidden, channels
    )
    layer_weights = {
        "ffn_fc1_weight": fc1_weight,
        "ffn_fc1_bias": fc1_bias,
        "ffn_fc2_weight": fc2_weight,
        "ffn_fc2_bias": fc2_bias,
    }
    for index in range(3):
        layer_weights[f"norm{index}_weight"] = _tensor(
            weights, f"{base}.norms.{index}.weight", (channels,)
        ).copy()
        layer_weights[f"norm{index}_bias"] = _tensor(
            weights, f"{base}.norms.{index}.bias", (channels,)
        ).copy()
    return {"temporal": temporal, "spatial": spatial, "layer": layer_weights}


def extract_decoder_cross_attention_weights(
    weights: Mapping[str, np.ndarray],
    layer: int,
    config: DecoderCrossAttentionConfig | None = None,
    *,
    decoder_prefix: str = DECODER_PREFIX,
) -> dict[str, np.ndarray]:
    """Select one official decoder CustomMSDeformableAttention parameter set."""

    weights = _mapping(weights)
    layer = _layer_index(layer)
    config = config or DecoderCrossAttentionConfig()
    if not isinstance(config, DecoderCrossAttentionConfig):
        raise ValidationError("config must be a DecoderCrossAttentionConfig")
    if not isinstance(decoder_prefix, str) or not decoder_prefix:
        raise ValidationError("decoder_prefix must be non-empty")
    base = f"{decoder_prefix}.{layer}.attentions.1"
    shapes = decoder_cross_attention_weight_shapes(config)
    result: dict[str, np.ndarray] = {}
    for canonical, module in (
        ("value", "value_proj"),
        ("offset", "sampling_offsets"),
        ("attention", "attention_weights"),
        ("output", "output_proj"),
    ):
        input_channels, output_channels = shapes[f"{canonical}_weight"]
        matrix, bias = _linear(
            weights,
            f"{base}.{module}",
            input_channels,
            output_channels,
        )
        result[f"{canonical}_weight"] = matrix
        result[f"{canonical}_bias"] = bias
    return result


def extract_decoder_self_attention_weights(
    weights: Mapping[str, np.ndarray],
    layer: int,
    config: DecoderSelfAttentionConfig | None = None,
    *,
    decoder_prefix: str = DECODER_PREFIX,
) -> dict[str, np.ndarray]:
    """Split MMCV/PyTorch packed QKV parameters for one decoder layer."""

    weights = _mapping(weights)
    layer = _layer_index(layer)
    config = config or DecoderSelfAttentionConfig()
    if not isinstance(config, DecoderSelfAttentionConfig):
        raise ValidationError("config must be a DecoderSelfAttentionConfig")
    if not isinstance(decoder_prefix, str) or not decoder_prefix:
        raise ValidationError("decoder_prefix must be non-empty")
    channels = config.embed_dim
    base = f"{decoder_prefix}.{layer}"
    attention = f"{base}.attentions.0.attn"
    packed_weight = _tensor(
        weights, f"{attention}.in_proj_weight", (channels * 3, channels)
    )
    packed_bias = _tensor(
        weights, f"{attention}.in_proj_bias", (channels * 3,)
    )
    result: dict[str, np.ndarray] = {}
    for index, canonical in enumerate(("q", "k", "v")):
        start, end = index * channels, (index + 1) * channels
        result[f"{canonical}_weight"] = np.ascontiguousarray(
            packed_weight[start:end].T
        )
        result[f"{canonical}_bias"] = packed_bias[start:end].copy()
    result["out_weight"], result["out_bias"] = _linear(
        weights, f"{attention}.out_proj", channels, channels
    )
    if config.post_layer_norm:
        norm = f"{base}.norms.0"
        result["norm_weight"] = _tensor(
            weights, f"{norm}.weight", (channels,)
        ).copy()
        result["norm_bias"] = _tensor(
            weights, f"{norm}.bias", (channels,)
        ).copy()
    return result


def extract_decoder_refinement_weights(
    weights: Mapping[str, np.ndarray],
    layer: int,
    config: DecoderReferenceRefinementConfig | None = None,
    *,
    head_prefix: str = "pts_bbox_head",
) -> dict[str, np.ndarray]:
    """Select and transpose one official decoder regression branch."""

    weights = _mapping(weights)
    layer = _layer_index(layer)
    config = config or DecoderReferenceRefinementConfig()
    if not isinstance(config, DecoderReferenceRefinementConfig):
        raise ValidationError(
            "config must be a DecoderReferenceRefinementConfig"
        )
    if not isinstance(head_prefix, str) or not head_prefix:
        raise ValidationError("head_prefix must be non-empty")
    base = f"{head_prefix}.reg_branches.{layer}"
    shapes = decoder_refinement_weight_shapes(config)
    result: dict[str, np.ndarray] = {}
    for canonical, module in (("fc0", "0"), ("fc1", "2"), ("fc2", "4")):
        input_channels, output_channels = shapes[f"{canonical}_weight"]
        matrix, bias = _linear(
            weights,
            f"{base}.{module}",
            input_channels,
            output_channels,
        )
        result[f"{canonical}_weight"] = matrix
        result[f"{canonical}_bias"] = bias
    return result


def extract_decoder_query_init_weights(
    weights: Mapping[str, np.ndarray],
    config: DecoderQueryInitConfig | None = None,
    *,
    head_prefix: str = "pts_bbox_head",
) -> dict[str, np.ndarray]:
    """Map learned object queries and the initial-reference projection."""

    weights = _mapping(weights)
    config = config or DecoderQueryInitConfig()
    if not isinstance(config, DecoderQueryInitConfig):
        raise ValidationError("config must be a DecoderQueryInitConfig")
    if not isinstance(head_prefix, str) or not head_prefix:
        raise ValidationError("head_prefix must be non-empty")
    channels = config.embed_dim
    embedding = _tensor(
        weights,
        f"{head_prefix}.query_embedding.weight",
        (config.queries, channels * 2),
    ).copy()
    reference_weight, reference_bias = _linear(
        weights,
        f"{head_prefix}.transformer.reference_points",
        channels,
        3,
    )
    return {
        "query_embedding": embedding,
        "reference_weight": reference_weight,
        "reference_bias": reference_bias,
    }


def extract_decoder_layer_weights(
    weights: Mapping[str, np.ndarray],
    layer: int,
    config: BevFormerDecoderLayerConfig | None = None,
    *,
    decoder_prefix: str = DECODER_PREFIX,
) -> dict[str, dict[str, np.ndarray]]:
    """Map every learned tensor used by one complete decoder layer graph."""

    weights = _mapping(weights)
    layer = _layer_index(layer)
    config = config or BevFormerDecoderLayerConfig()
    if not isinstance(config, BevFormerDecoderLayerConfig):
        raise ValidationError("config must be a BevFormerDecoderLayerConfig")
    if not isinstance(decoder_prefix, str) or not decoder_prefix:
        raise ValidationError("decoder_prefix must be non-empty")

    self_attention = extract_decoder_self_attention_weights(
        weights,
        layer,
        config.self_attention,
        decoder_prefix=decoder_prefix,
    )
    cross_attention = extract_decoder_cross_attention_weights(
        weights,
        layer,
        config.cross_attention,
        decoder_prefix=decoder_prefix,
    )
    base = f"{decoder_prefix}.{layer}"
    channels = config.embed_dim
    hidden = config.ffn.hidden_dim
    fc1_weight, fc1_bias = _linear(
        weights, f"{base}.ffns.0.layers.0.0", channels, hidden
    )
    fc2_weight, fc2_bias = _linear(
        weights, f"{base}.ffns.0.layers.1", hidden, channels
    )
    layer_weights = {
        "norm1_weight": _tensor(
            weights, f"{base}.norms.1.weight", (channels,)
        ).copy(),
        "norm1_bias": _tensor(
            weights, f"{base}.norms.1.bias", (channels,)
        ).copy(),
        "ffn_fc1_weight": fc1_weight,
        "ffn_fc1_bias": fc1_bias,
        "ffn_fc2_weight": fc2_weight,
        "ffn_fc2_bias": fc2_bias,
        "norm2_weight": _tensor(
            weights, f"{base}.norms.2.weight", (channels,)
        ).copy(),
        "norm2_bias": _tensor(
            weights, f"{base}.norms.2.bias", (channels,)
        ).copy(),
    }
    return {
        "self_attention": self_attention,
        "cross_attention": cross_attention,
        "layer": layer_weights,
    }


def extract_detection_head_weights(
    weights: Mapping[str, np.ndarray],
    layer: int,
    config: DetectionHeadConfig | None = None,
    *,
    head_prefix: str = "pts_bbox_head",
) -> dict[str, np.ndarray]:
    """Select one official classification branch for raw detection outputs.

    Regression tensors are supplied at runtime because the detection graph
    consumes the output of the matching ``reg_branches[layer]`` refiner.  The
    NMS-free top-k decode remains outside this graph.
    """

    weights = _mapping(weights)
    layer = _layer_index(layer)
    config = config or DetectionHeadConfig()
    if not isinstance(config, DetectionHeadConfig):
        raise ValidationError("config must be a DetectionHeadConfig")
    if not isinstance(head_prefix, str) or not head_prefix:
        raise ValidationError("head_prefix must be non-empty")
    base = f"{head_prefix}.cls_branches.{layer}"
    channels = config.embed_dim
    result: dict[str, np.ndarray] = {}
    for canonical, module, input_channels, output_channels in (
        ("cls0", "0", channels, channels),
        ("cls1", "3", channels, channels),
        ("cls2", "6", channels, config.classes),
    ):
        matrix, bias = _linear(
            weights,
            f"{base}.{module}",
            input_channels,
            output_channels,
        )
        result[f"{canonical}_weight"] = matrix
        result[f"{canonical}_bias"] = bias
    for canonical, module in (("cls0", "1"), ("cls1", "4")):
        result[f"{canonical}_norm_weight"] = _tensor(
            weights, f"{base}.{module}.weight", (channels,)
        ).copy()
        result[f"{canonical}_norm_bias"] = _tensor(
            weights, f"{base}.{module}.bias", (channels,)
        ).copy()
    return result


def extract_feature_prep_weights(
    weights: Mapping[str, np.ndarray],
    config: BevFormerFeaturePrepConfig | None = None,
    *,
    transformer_prefix: str = "pts_bbox_head.transformer",
) -> dict[str, np.ndarray]:
    """Map official camera and single-level feature embeddings."""

    weights = _mapping(weights)
    config = config or BevFormerFeaturePrepConfig()
    if not isinstance(config, BevFormerFeaturePrepConfig):
        raise ValidationError("config must be a BevFormerFeaturePrepConfig")
    if not isinstance(transformer_prefix, str) or not transformer_prefix:
        raise ValidationError("transformer_prefix must be non-empty")
    target_shapes = feature_prep_weight_shapes(config)
    return {
        "camera_embeddings": _tensor(
            weights,
            f"{transformer_prefix}.cams_embeds",
            target_shapes["camera_embeddings"],
        ).copy(),
        "level_embedding": _leading_rows_tensor(
            weights,
            f"{transformer_prefix}.level_embeds",
            target_shapes["level_embedding"],
            allowed_source_rows=(target_shapes["level_embedding"][0], 4),
        ).copy(),
    }


def extract_encoder_input_weights(
    weights: Mapping[str, np.ndarray],
    config: BevFormerEncoderInputConfig | None = None,
    *,
    head_prefix: str = "pts_bbox_head",
) -> dict[str, np.ndarray]:
    """Map learned BEV queries and CAN-bus conditioning parameters."""

    weights = _mapping(weights)
    config = config or BevFormerEncoderInputConfig()
    if not isinstance(config, BevFormerEncoderInputConfig):
        raise ValidationError("config must be a BevFormerEncoderInputConfig")
    if not isinstance(head_prefix, str) or not head_prefix:
        raise ValidationError("head_prefix must be non-empty")
    transformer = f"{head_prefix}.transformer"
    fc0_weight, fc0_bias = _linear(
        weights,
        f"{transformer}.can_bus_mlp.0",
        config.can_bus_dim,
        config.can_bus_hidden_dim,
    )
    fc1_weight, fc1_bias = _linear(
        weights,
        f"{transformer}.can_bus_mlp.2",
        config.can_bus_hidden_dim,
        config.embed_dim,
    )
    return {
        "bev_embedding": _tensor(
            weights,
            f"{head_prefix}.bev_embedding.weight",
            (config.queries, config.embed_dim),
        ).copy(),
        "row_embedding": _tensor(
            weights,
            f"{head_prefix}.positional_encoding.row_embed.weight",
            (config.bev_h, config.embed_dim // 2),
        ).copy(),
        "col_embedding": _tensor(
            weights,
            f"{head_prefix}.positional_encoding.col_embed.weight",
            (config.bev_w, config.embed_dim // 2),
        ).copy(),
        "can_bus_fc0_weight": fc0_weight,
        "can_bus_fc0_bias": fc0_bias,
        "can_bus_fc1_weight": fc1_weight,
        "can_bus_fc1_bias": fc1_bias,
        "can_bus_norm_weight": _tensor(
            weights,
            f"{transformer}.can_bus_mlp.norm.weight",
            (config.embed_dim,),
        ).copy(),
        "can_bus_norm_bias": _tensor(
            weights,
            f"{transformer}.can_bus_mlp.norm.bias",
            (config.embed_dim,),
        ).copy(),
    }


def extract_decoder_stack_weights(
    weights: Mapping[str, np.ndarray],
    config: BevFormerDecoderStackConfig | None = None,
    *,
    decoder_prefix: str = DECODER_PREFIX,
    head_prefix: str = "pts_bbox_head",
) -> dict[str, object]:
    """Map learned initialization and all six official decoder layers."""

    weights = _mapping(weights)
    config = config or BevFormerDecoderStackConfig()
    if not isinstance(config, BevFormerDecoderStackConfig):
        raise ValidationError("config must be a BevFormerDecoderStackConfig")
    query_init = extract_decoder_query_init_weights(
        weights,
        config.query_init,
        head_prefix=head_prefix,
    )
    layers = [
        extract_decoder_layer_weights(
            weights,
            index,
            config.layer,
            decoder_prefix=decoder_prefix,
        )
        for index in range(config.num_layers)
    ]
    refinement = tuple(
        extract_decoder_refinement_weights(
            weights,
            index,
            config.refinement,
            head_prefix=head_prefix,
        )
        for index in range(config.num_layers)
    )
    return {
        "query_init": query_init,
        "self_attention": tuple(group["self_attention"] for group in layers),
        "cross_attention": tuple(group["cross_attention"] for group in layers),
        "layer": tuple(group["layer"] for group in layers),
        "refinement": refinement,
    }


def extract_detection_stack_weights(
    weights: Mapping[str, np.ndarray],
    config: DetectionStackConfig | None = None,
    *,
    head_prefix: str = "pts_bbox_head",
) -> tuple[dict[str, np.ndarray], ...]:
    """Map the six independent official classification branches."""

    weights = _mapping(weights)
    config = config or DetectionStackConfig()
    if not isinstance(config, DetectionStackConfig):
        raise ValidationError("config must be a DetectionStackConfig")
    return tuple(
        extract_detection_head_weights(
            weights,
            index,
            config.head,
            head_prefix=head_prefix,
        )
        for index in range(config.num_layers)
    )


__all__ = [
    "DECODER_PREFIX",
    "ENCODER_PREFIX",
    "extract_decoder_cross_attention_weights",
    "extract_decoder_layer_weights",
    "extract_decoder_refinement_weights",
    "extract_decoder_query_init_weights",
    "extract_decoder_self_attention_weights",
    "extract_decoder_stack_weights",
    "extract_detection_head_weights",
    "extract_detection_stack_weights",
    "extract_encoder_ffn_weights",
    "extract_encoder_input_weights",
    "extract_encoder_layer_weights",
    "extract_feature_prep_weights",
    "extract_spatial_attention_weights",
    "extract_temporal_attention_weights",
]
