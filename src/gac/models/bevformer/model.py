"""BEVFormer-tiny contracts used by the first compiler reference slice.

The initial slice isolates the multi-scale deformable attention (MSDA)
invocation inside BEVFormer's spatial cross-attention.  BEVFormer folds the
camera dimension into the batch dimension before calling MSDA, so a profile
with batch one and six cameras produces an effective MSDA batch of six.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Any

import numpy as np

from gac.core.errors import ValidationError
from gac.core.ir import Graph, Node, TensorSpec
from gac.compiler.passes import ShapeInferencePass


@dataclass(frozen=True, slots=True)
class BevFormerTinyProfile:
    """Static BEVFormer-tiny dimensions for the SM86 reference slice."""

    batch: int = 1
    cameras: int = 6
    image_h: int = 480
    image_w: int = 800
    bev_h: int = 50
    bev_w: int = 50
    embed_dim: int = 256
    heads: int = 8
    feature_shapes: tuple[tuple[int, int], ...] = ((15, 25),)
    points: int = 8
    encoder_layers: int = 3
    decoder_layers: int = 6
    object_queries: int = 900
    queue_length: int = 3

    def __post_init__(self) -> None:
        scalar_fields = (
            "batch",
            "cameras",
            "image_h",
            "image_w",
            "bev_h",
            "bev_w",
            "embed_dim",
            "heads",
            "points",
            "encoder_layers",
            "decoder_layers",
            "object_queries",
            "queue_length",
        )
        for name in scalar_fields:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise ValidationError(f"BEVFormer profile {name} must be a positive integer")
            normalized = int(value)
            if normalized <= 0:
                raise ValidationError(f"BEVFormer profile {name} must be positive")
            object.__setattr__(self, name, normalized)

        try:
            feature_shapes = tuple(tuple(shape) for shape in self.feature_shapes)
        except TypeError as exc:
            raise ValidationError(
                "BEVFormer profile feature_shapes must be a non-empty sequence of (H, W) pairs"
            ) from exc
        if not feature_shapes:
            raise ValidationError("BEVFormer profile feature_shapes must not be empty")
        normalized_shapes: list[tuple[int, int]] = []
        for level, shape in enumerate(feature_shapes):
            if len(shape) != 2:
                raise ValidationError(
                    f"BEVFormer feature level {level} must contain exactly (H, W)"
                )
            height, width = shape
            if (
                isinstance(height, bool)
                or isinstance(width, bool)
                or not isinstance(height, Integral)
                or not isinstance(width, Integral)
                or int(height) <= 0
                or int(width) <= 0
            ):
                raise ValidationError(
                    f"BEVFormer feature level {level} dimensions must be positive integers"
                )
            normalized_shapes.append((int(height), int(width)))
        object.__setattr__(self, "feature_shapes", tuple(normalized_shapes))

        if self.embed_dim % self.heads != 0:
            raise ValidationError(
                "BEVFormer profile embed_dim must be divisible by heads"
            )

    @property
    def head_dim(self) -> int:
        return self.embed_dim // self.heads

    @property
    def feature_elements(self) -> int:
        return sum(height * width for height, width in self.feature_shapes)

    @property
    def bev_queries(self) -> int:
        return self.bev_h * self.bev_w

    @property
    def msda_batch(self) -> int:
        return self.batch * self.cameras

    def metadata(self) -> dict[str, Any]:
        """Return a JSON-compatible description without derived dimensions."""

        return {
            "batch": self.batch,
            "cameras": self.cameras,
            "image_h": self.image_h,
            "image_w": self.image_w,
            "bev_h": self.bev_h,
            "bev_w": self.bev_w,
            "embed_dim": self.embed_dim,
            "heads": self.heads,
            "feature_shapes": [list(shape) for shape in self.feature_shapes],
            "points": self.points,
            "encoder_layers": self.encoder_layers,
            "decoder_layers": self.decoder_layers,
            "object_queries": self.object_queries,
            "queue_length": self.queue_length,
        }


def build_msda_graph(
    profile: BevFormerTinyProfile = BevFormerTinyProfile(),
) -> Graph:
    """Build and infer the FP32 BEVFormer-tiny spatial MSDA graph."""

    if not isinstance(profile, BevFormerTinyProfile):
        raise ValidationError("profile must be a BevFormerTinyProfile")

    levels = len(profile.feature_shapes)
    batch = profile.msda_batch
    queries = profile.bev_queries
    inputs = {
        "value": TensorSpec(
            (batch, profile.feature_elements, profile.heads, profile.head_dim),
            "float32",
        ),
        "sampling_locations": TensorSpec(
            (batch, queries, profile.heads, levels, profile.points, 2),
            "float32",
        ),
        "attention_weights": TensorSpec(
            (batch, queries, profile.heads, levels, profile.points),
            "float32",
        ),
    }
    graph = Graph(
        name="bevformer_tiny_msda",
        inputs=inputs,
        outputs=("output",),
        nodes=[
            Node(
                id="msda",
                op="MSDeformableAttention",
                inputs=("value", "sampling_locations", "attention_weights"),
                outputs=("output",),
                attrs={
                    "spatial_shapes": [list(shape) for shape in profile.feature_shapes]
                },
            )
        ],
        metadata={
            "profile": profile.metadata(),
            "reference_slice": {
                "name": "bevformer_tiny_spatial_cross_attention_msda",
                "component": "SpatialCrossAttention/MSDeformableAttention3D",
                "kind": "fp32-reference-slice",
                "dtype": "float32",
                "effective_batch": "batch*cameras",
            },
        },
    )
    ShapeInferencePass().run(graph)
    graph.validate()
    return graph


def make_msda_inputs(graph: Graph, seed: int = 7) -> dict[str, np.ndarray]:
    """Create deterministic, contiguous FP32 inputs for an MSDA graph."""

    if isinstance(seed, bool) or not isinstance(seed, Integral):
        raise ValidationError("MSDA input seed must be an integer")
    required = ("value", "sampling_locations", "attention_weights")
    missing = [name for name in required if name not in graph.inputs]
    if missing:
        raise ValidationError(f"MSDA graph is missing inputs: {missing}")

    shapes: dict[str, tuple[int, ...]] = {}
    for name in required:
        spec = graph.inputs[name]
        if not spec.is_static:
            raise ValidationError(f"MSDA input {name!r} must have a static shape")
        if spec.dtype != "float32":
            raise ValidationError(f"MSDA input {name!r} must have dtype float32")
        shapes[name] = tuple(int(dim) for dim in spec.shape)

    rng = np.random.default_rng(int(seed))
    value = rng.normal(size=shapes["value"]).astype(np.float32)
    sampling_locations = rng.uniform(
        -0.25,
        1.25,
        size=shapes["sampling_locations"],
    ).astype(np.float32)
    attention_weights = rng.uniform(
        np.float32(0.01),
        np.float32(1.0),
        size=shapes["attention_weights"],
    ).astype(np.float32)
    normalizer = attention_weights.sum(axis=(-2, -1), keepdims=True, dtype=np.float32)
    attention_weights = attention_weights / normalizer

    return {
        "value": np.ascontiguousarray(value, dtype=np.float32),
        "sampling_locations": np.ascontiguousarray(
            sampling_locations, dtype=np.float32
        ),
        "attention_weights": np.ascontiguousarray(
            attention_weights, dtype=np.float32
        ),
    }


__all__ = ["BevFormerTinyProfile", "build_msda_graph", "make_msda_inputs"]
