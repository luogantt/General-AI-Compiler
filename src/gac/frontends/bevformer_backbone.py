"""Framework-independent BEVFormer ResNet-50 and single-level FPN importer.

The importer consumes a mapping of MMDetection-style checkpoint names to
NumPy arrays.  It does not import PyTorch or MMCV and does not execute the
network while constructing the canonical graph.  All parameter names are
centralized in :class:`MmdetCheckpointKeyMap` so a checkpoint naming change
fails with a precise key diagnostic instead of silently changing topology.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral
from typing import Any

import numpy as np

from gac.ir import Graph, Node, TensorSpec
from gac.passes import ShapeInferencePass


_ALLOWED_NODE_OPS = frozenset(
    {"Conv2D", "BatchNorm", "ReLU", "Add", "MaxPool2D", "Cast"}
)


class BevFormerBackboneError(ValueError):
    """The profile or checkpoint cannot form the canonical backbone graph."""


def _positive_integer(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) <= 0:
        raise BevFormerBackboneError(f"{name} must be a positive integer")
    return int(value)


def _floating_dtype(value: Any, *, name: str) -> str:
    try:
        dtype = np.dtype(value)
    except TypeError as exc:
        raise BevFormerBackboneError(f"{name} must be a floating dtype") from exc
    if dtype.name not in {"float16", "float32"}:
        raise BevFormerBackboneError(
            f"{name} must be float16 or float32, got {dtype.name}"
        )
    return dtype.name


@dataclass(frozen=True, slots=True)
class ResNetFpnProfile:
    """Static topology and dimensions for a bottleneck ResNet plus one FPN level."""

    name: str
    batch: int
    input_channels: int
    input_hw: tuple[int, int]
    stem_channels: int
    stage_planes: tuple[int, int, int, int]
    blocks: tuple[int, int, int, int]
    expansion: int
    fpn_channels: int
    input_dtype: str = "float16"
    parameter_dtype: str = "float32"
    batch_norm_epsilon: float = 1e-5

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise BevFormerBackboneError("profile name must be a non-empty string")
        for field_name in (
            "batch",
            "input_channels",
            "stem_channels",
            "expansion",
            "fpn_channels",
        ):
            object.__setattr__(
                self,
                field_name,
                _positive_integer(getattr(self, field_name), name=field_name),
            )
        if not isinstance(self.input_hw, tuple) or len(self.input_hw) != 2:
            raise BevFormerBackboneError("input_hw must be a (height, width) tuple")
        object.__setattr__(
            self,
            "input_hw",
            tuple(
                _positive_integer(value, name="input_hw dimension")
                for value in self.input_hw
            ),
        )
        for field_name in ("stage_planes", "blocks"):
            raw = getattr(self, field_name)
            if not isinstance(raw, tuple) or len(raw) != 4:
                raise BevFormerBackboneError(f"{field_name} must contain four values")
            object.__setattr__(
                self,
                field_name,
                tuple(
                    _positive_integer(value, name=f"{field_name} value")
                    for value in raw
                ),
            )
        object.__setattr__(
            self,
            "input_dtype",
            _floating_dtype(self.input_dtype, name="input_dtype"),
        )
        object.__setattr__(
            self,
            "parameter_dtype",
            _floating_dtype(self.parameter_dtype, name="parameter_dtype"),
        )
        if self.parameter_dtype != "float32":
            raise BevFormerBackboneError(
                "checkpoint parameter_dtype must be float32 for the first profile"
            )
        if (
            isinstance(self.batch_norm_epsilon, bool)
            or not isinstance(self.batch_norm_epsilon, (int, float))
            or not math.isfinite(float(self.batch_norm_epsilon))
            or float(self.batch_norm_epsilon) <= 0
        ):
            raise BevFormerBackboneError(
                "batch_norm_epsilon must be finite and positive"
            )
        object.__setattr__(
            self, "batch_norm_epsilon", float(self.batch_norm_epsilon)
        )

    @property
    def input_shape(self) -> tuple[int, int, int, int]:
        return (self.batch, self.input_channels, *self.input_hw)

    @property
    def c5_channels(self) -> int:
        return self.stage_planes[-1] * self.expansion

    @property
    def output_hw(self) -> tuple[int, int]:
        # Stem convolution, max-pool, and the first blocks of stages 2-4 each
        # downsample with padding that implements ceil(size / 2).
        return tuple((dimension + 31) // 32 for dimension in self.input_hw)  # type: ignore[return-value]

    @property
    def output_shape(self) -> tuple[int, int, int, int]:
        return (self.batch, self.fpn_channels, *self.output_hw)

    def to_metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "batch": self.batch,
            "input_channels": self.input_channels,
            "input_hw": list(self.input_hw),
            "stem_channels": self.stem_channels,
            "stage_planes": list(self.stage_planes),
            "blocks": list(self.blocks),
            "expansion": self.expansion,
            "fpn_channels": self.fpn_channels,
            "input_dtype": self.input_dtype,
            "parameter_dtype": self.parameter_dtype,
            "batch_norm_epsilon": self.batch_norm_epsilon,
            "output_shape": list(self.output_shape),
        }


BEVFORMER_R50_FPN_PROFILE = ResNetFpnProfile(
    name="bevformer-tiny-r50-c5-fpn256-b6-480x800-v1",
    batch=6,
    input_channels=3,
    input_hw=(480, 800),
    stem_channels=64,
    stage_planes=(64, 128, 256, 512),
    blocks=(3, 4, 6, 3),
    expansion=4,
    fpn_channels=256,
)


@dataclass(frozen=True, slots=True)
class MmdetCheckpointKeyMap:
    """Central definition of the MMDetection state-dict naming convention."""

    backbone_prefix: str = "img_backbone"
    neck_prefix: str = "img_neck"

    def __post_init__(self) -> None:
        for name, value in (
            ("backbone_prefix", self.backbone_prefix),
            ("neck_prefix", self.neck_prefix),
        ):
            if not isinstance(value, str) or not value or value.endswith("."):
                raise BevFormerBackboneError(
                    f"checkpoint {name} must be a non-empty prefix without trailing dot"
                )

    def stem_conv(self) -> str:
        return f"{self.backbone_prefix}.conv1"

    def stem_bn(self) -> str:
        return f"{self.backbone_prefix}.bn1"

    def block(self, stage: int, block: int) -> str:
        return f"{self.backbone_prefix}.layer{stage}.{block}"

    def block_conv(self, stage: int, block: int, convolution: int) -> str:
        return f"{self.block(stage, block)}.conv{convolution}"

    def block_bn(self, stage: int, block: int, normalization: int) -> str:
        return f"{self.block(stage, block)}.bn{normalization}"

    def downsample_conv(self, stage: int, block: int) -> str:
        return f"{self.block(stage, block)}.downsample.0"

    def downsample_bn(self, stage: int, block: int) -> str:
        return f"{self.block(stage, block)}.downsample.1"

    def lateral_conv(self) -> str:
        return f"{self.neck_prefix}.lateral_convs.0.conv"

    def fpn_conv(self) -> str:
        return f"{self.neck_prefix}.fpn_convs.0.conv"

    @staticmethod
    def weight(module: str) -> str:
        return f"{module}.weight"

    @staticmethod
    def bias(module: str) -> str:
        return f"{module}.bias"

    @staticmethod
    def running_mean(module: str) -> str:
        return f"{module}.running_mean"

    @staticmethod
    def running_var(module: str) -> str:
        return f"{module}.running_var"


MMDET_CHECKPOINT_KEYS = MmdetCheckpointKeyMap()


@dataclass(frozen=True, slots=True)
class CheckpointTensorSpec:
    """One required checkpoint tensor and its exact static contract."""

    key: str
    shape: tuple[int, ...]
    dtype: str
    role: str

    def __post_init__(self) -> None:
        if not isinstance(self.key, str) or not self.key:
            raise BevFormerBackboneError("checkpoint spec key must be non-empty")
        if not isinstance(self.shape, tuple) or not self.shape:
            raise BevFormerBackboneError(
                f"checkpoint spec {self.key!r} needs a non-empty shape"
            )
        for dimension in self.shape:
            _positive_integer(dimension, name=f"{self.key} shape dimension")
        _floating_dtype(self.dtype, name=f"{self.key} dtype")
        if not isinstance(self.role, str) or not self.role:
            raise BevFormerBackboneError(
                f"checkpoint spec {self.key!r} needs a role"
            )


def _conv_spec(
    module: str,
    out_channels: int,
    in_channels: int,
    kernel: int,
    *,
    dtype: str,
    key_map: MmdetCheckpointKeyMap,
    bias: bool = False,
) -> list[CheckpointTensorSpec]:
    specs = [
        CheckpointTensorSpec(
            key_map.weight(module),
            (out_channels, in_channels, kernel, kernel),
            dtype,
            "conv-weight",
        )
    ]
    if bias:
        specs.append(
            CheckpointTensorSpec(
                key_map.bias(module), (out_channels,), dtype, "conv-bias"
            )
        )
    return specs


def _batch_norm_specs(
    module: str,
    channels: int,
    *,
    dtype: str,
    key_map: MmdetCheckpointKeyMap,
) -> list[CheckpointTensorSpec]:
    return [
        CheckpointTensorSpec(key_map.weight(module), (channels,), dtype, "bn-scale"),
        CheckpointTensorSpec(key_map.bias(module), (channels,), dtype, "bn-bias"),
        CheckpointTensorSpec(
            key_map.running_mean(module), (channels,), dtype, "bn-running-mean"
        ),
        CheckpointTensorSpec(
            key_map.running_var(module), (channels,), dtype, "bn-running-variance"
        ),
    ]


def required_checkpoint_specs(
    profile: ResNetFpnProfile = BEVFORMER_R50_FPN_PROFILE,
    *,
    key_map: MmdetCheckpointKeyMap = MMDET_CHECKPOINT_KEYS,
) -> tuple[CheckpointTensorSpec, ...]:
    """Return every required tensor in deterministic graph-consumption order."""

    if not isinstance(profile, ResNetFpnProfile):
        raise BevFormerBackboneError("profile must be a ResNetFpnProfile")
    if not isinstance(key_map, MmdetCheckpointKeyMap):
        raise BevFormerBackboneError("key_map must be a MmdetCheckpointKeyMap")
    dtype = profile.parameter_dtype
    specs: list[CheckpointTensorSpec] = []
    specs.extend(
        _conv_spec(
            key_map.stem_conv(),
            profile.stem_channels,
            profile.input_channels,
            7,
            dtype=dtype,
            key_map=key_map,
        )
    )
    specs.extend(
        _batch_norm_specs(
            key_map.stem_bn(),
            profile.stem_channels,
            dtype=dtype,
            key_map=key_map,
        )
    )

    in_channels = profile.stem_channels
    for stage_index, (planes, block_count) in enumerate(
        zip(profile.stage_planes, profile.blocks, strict=True), start=1
    ):
        out_channels = planes * profile.expansion
        for block_index in range(block_count):
            stride = 1 if stage_index == 1 or block_index > 0 else 2
            for convolution, (channels_in, channels_out, kernel) in enumerate(
                (
                    (in_channels, planes, 1),
                    (planes, planes, 3),
                    (planes, out_channels, 1),
                ),
                start=1,
            ):
                specs.extend(
                    _conv_spec(
                        key_map.block_conv(
                            stage_index, block_index, convolution
                        ),
                        channels_out,
                        channels_in,
                        kernel,
                        dtype=dtype,
                        key_map=key_map,
                    )
                )
                specs.extend(
                    _batch_norm_specs(
                        key_map.block_bn(
                            stage_index, block_index, convolution
                        ),
                        channels_out,
                        dtype=dtype,
                        key_map=key_map,
                    )
                )
            if block_index == 0 and (stride != 1 or in_channels != out_channels):
                specs.extend(
                    _conv_spec(
                        key_map.downsample_conv(stage_index, block_index),
                        out_channels,
                        in_channels,
                        1,
                        dtype=dtype,
                        key_map=key_map,
                    )
                )
                specs.extend(
                    _batch_norm_specs(
                        key_map.downsample_bn(stage_index, block_index),
                        out_channels,
                        dtype=dtype,
                        key_map=key_map,
                    )
                )
            in_channels = out_channels

    specs.extend(
        _conv_spec(
            key_map.lateral_conv(),
            profile.fpn_channels,
            profile.c5_channels,
            1,
            dtype=dtype,
            key_map=key_map,
            bias=True,
        )
    )
    specs.extend(
        _conv_spec(
            key_map.fpn_conv(),
            profile.fpn_channels,
            profile.fpn_channels,
            3,
            dtype=dtype,
            key_map=key_map,
            bias=True,
        )
    )
    keys = [spec.key for spec in specs]
    if len(keys) != len(set(keys)):
        raise BevFormerBackboneError("checkpoint key map generated duplicate keys")
    return tuple(specs)


def validate_checkpoint_weights(
    weights: Mapping[str, np.ndarray],
    profile: ResNetFpnProfile = BEVFORMER_R50_FPN_PROFILE,
    *,
    key_map: MmdetCheckpointKeyMap = MMDET_CHECKPOINT_KEYS,
) -> dict[str, np.ndarray]:
    """Validate and select required NumPy weights.

    Unrelated detector, transformer, optimizer, and metadata entries are
    permitted and omitted from the returned mapping.  Required arrays must be
    exact, finite, C-contiguous float32 tensors.
    """

    if not isinstance(weights, Mapping):
        raise BevFormerBackboneError(
            "checkpoint weights must be a string-to-NumPy-array mapping"
        )
    invalid_keys = [key for key in weights if not isinstance(key, str)]
    if invalid_keys:
        raise BevFormerBackboneError("checkpoint keys must all be strings")
    specs = required_checkpoint_specs(profile, key_map=key_map)
    missing = [spec.key for spec in specs if spec.key not in weights]
    if missing:
        module_prefixed = all(f"module.{key}" in weights for key in missing)
        preview = ", ".join(repr(key) for key in missing[:5])
        suffix = "" if len(missing) <= 5 else f", ... ({len(missing)} total)"
        hint = (
            " The checkpoint appears to use a 'module.' prefix; strip that prefix first."
            if module_prefixed
            else ""
        )
        raise BevFormerBackboneError(
            f"checkpoint is missing required tensors: {preview}{suffix}.{hint}"
        )

    selected: dict[str, np.ndarray] = {}
    for spec in specs:
        value = weights[spec.key]
        if not isinstance(value, np.ndarray):
            raise BevFormerBackboneError(
                f"checkpoint tensor {spec.key!r} must be a NumPy array, "
                f"got {type(value).__name__}"
            )
        if value.shape != spec.shape:
            raise BevFormerBackboneError(
                f"checkpoint tensor {spec.key!r} has shape {value.shape}; "
                f"expected {spec.shape}"
            )
        if value.dtype.name != spec.dtype:
            raise BevFormerBackboneError(
                f"checkpoint tensor {spec.key!r} has dtype {value.dtype.name}; "
                f"expected {spec.dtype}"
            )
        if not value.flags.c_contiguous:
            raise BevFormerBackboneError(
                f"checkpoint tensor {spec.key!r} must be C-contiguous"
            )
        if not np.isfinite(value).all():
            raise BevFormerBackboneError(
                f"checkpoint tensor {spec.key!r} contains NaN or infinity"
            )
        selected[spec.key] = value
    return selected


class _GraphBuilder:
    def __init__(
        self,
        profile: ResNetFpnProfile,
        key_map: MmdetCheckpointKeyMap,
    ) -> None:
        self.profile = profile
        self.key_map = key_map
        self.nodes: list[Node] = []

    @staticmethod
    def _id(module: str, suffix: str = "") -> str:
        base = module.replace(".", "_")
        return f"{base}_{suffix}" if suffix else base

    def cast(self, value: str) -> str:
        if self.profile.input_dtype == self.profile.parameter_dtype:
            return value
        output = "images_compute"
        self.nodes.append(
            Node(
                "images_to_compute_dtype",
                "Cast",
                (value,),
                (output,),
                {"to": self.profile.parameter_dtype},
                source_nodes=("bevformer_export.images.cast",),
            )
        )
        return output

    def conv(
        self,
        value: str,
        module: str,
        *,
        stride: int = 1,
        padding: int = 0,
        bias: bool = False,
    ) -> str:
        output = self._id(module, "out")
        inputs = [value, self.key_map.weight(module)]
        if bias:
            inputs.append(self.key_map.bias(module))
        self.nodes.append(
            Node(
                self._id(module),
                "Conv2D",
                tuple(inputs),
                (output,),
                {
                    "strides": [stride, stride],
                    "pads": [padding, padding, padding, padding],
                    "dilations": [1, 1],
                    "groups": 1,
                },
                source_nodes=(module,),
            )
        )
        return output

    def batch_norm(self, value: str, module: str) -> str:
        output = self._id(module, "out")
        self.nodes.append(
            Node(
                self._id(module),
                "BatchNorm",
                (
                    value,
                    self.key_map.weight(module),
                    self.key_map.bias(module),
                    self.key_map.running_mean(module),
                    self.key_map.running_var(module),
                ),
                (output,),
                {"epsilon": self.profile.batch_norm_epsilon},
                source_nodes=(module,),
            )
        )
        return output

    def relu(self, value: str, source: str, suffix: str) -> str:
        node_id = self._id(source, suffix)
        output = f"{node_id}_out"
        self.nodes.append(
            Node(
                node_id,
                "ReLU",
                (value,),
                (output,),
                source_nodes=(f"{source}.relu",),
            )
        )
        return output

    def add(self, left: str, right: str, source: str) -> str:
        node_id = self._id(source, "residual_add")
        output = f"{node_id}_out"
        self.nodes.append(
            Node(
                node_id,
                "Add",
                (left, right),
                (output,),
                source_nodes=(source,),
            )
        )
        return output

    def max_pool(self, value: str) -> str:
        output = "img_backbone_maxpool_out"
        self.nodes.append(
            Node(
                "img_backbone_maxpool",
                "MaxPool2D",
                (value,),
                (output,),
                {
                    "kernel_shape": [3, 3],
                    "strides": [2, 2],
                    "pads": [1, 1, 1, 1],
                    "dilations": [1, 1],
                    "ceil_mode": False,
                },
                source_nodes=(f"{self.key_map.backbone_prefix}.maxpool",),
            )
        )
        return output


def build_bevformer_backbone_graph(
    weights: Mapping[str, np.ndarray],
    profile: ResNetFpnProfile = BEVFORMER_R50_FPN_PROFILE,
    *,
    key_map: MmdetCheckpointKeyMap = MMDET_CHECKPOINT_KEYS,
) -> Graph:
    """Build and shape-infer the ResNet bottlenecks and the C5-only FPN."""

    if not isinstance(profile, ResNetFpnProfile):
        raise BevFormerBackboneError("profile must be a ResNetFpnProfile")
    if not isinstance(key_map, MmdetCheckpointKeyMap):
        raise BevFormerBackboneError("key_map must be a MmdetCheckpointKeyMap")
    constants = validate_checkpoint_weights(weights, profile, key_map=key_map)
    builder = _GraphBuilder(profile, key_map)

    current = builder.cast("images")
    current = builder.conv(
        current, key_map.stem_conv(), stride=2, padding=3
    )
    current = builder.batch_norm(current, key_map.stem_bn())
    current = builder.relu(current, key_map.backbone_prefix, "stem_relu")
    current = builder.max_pool(current)

    in_channels = profile.stem_channels
    for stage_index, (planes, block_count) in enumerate(
        zip(profile.stage_planes, profile.blocks, strict=True), start=1
    ):
        out_channels = planes * profile.expansion
        for block_index in range(block_count):
            block = key_map.block(stage_index, block_index)
            stride = 1 if stage_index == 1 or block_index > 0 else 2
            identity = current

            branch = builder.conv(
                current, key_map.block_conv(stage_index, block_index, 1)
            )
            branch = builder.batch_norm(
                branch, key_map.block_bn(stage_index, block_index, 1)
            )
            branch = builder.relu(branch, block, "conv1_relu")
            branch = builder.conv(
                branch,
                key_map.block_conv(stage_index, block_index, 2),
                stride=stride,
                padding=1,
            )
            branch = builder.batch_norm(
                branch, key_map.block_bn(stage_index, block_index, 2)
            )
            branch = builder.relu(branch, block, "conv2_relu")
            branch = builder.conv(
                branch, key_map.block_conv(stage_index, block_index, 3)
            )
            branch = builder.batch_norm(
                branch, key_map.block_bn(stage_index, block_index, 3)
            )

            if block_index == 0 and (stride != 1 or in_channels != out_channels):
                identity = builder.conv(
                    identity,
                    key_map.downsample_conv(stage_index, block_index),
                    stride=stride,
                )
                identity = builder.batch_norm(
                    identity, key_map.downsample_bn(stage_index, block_index)
                )
            current = builder.add(branch, identity, block)
            current = builder.relu(current, block, "output_relu")
            in_channels = out_channels

    current = builder.conv(current, key_map.lateral_conv(), bias=True)
    current = builder.conv(
        current, key_map.fpn_conv(), padding=1, bias=True
    )
    output_name = "fpn_c5"
    final_node = builder.nodes[-1]
    builder.nodes[-1] = Node(
        final_node.id,
        final_node.op,
        final_node.inputs,
        (output_name,),
        final_node.attrs,
        final_node.source_nodes,
    )

    required_keys = set(constants)
    ignored_keys = sorted(key for key in weights if key not in required_keys)
    graph = Graph(
        name="bevformer_r50_single_level_fpn",
        inputs={"images": TensorSpec(profile.input_shape, profile.input_dtype, "NCHW")},
        outputs=(output_name,),
        nodes=builder.nodes,
        constants=constants,
        metadata={
            "component": "BEVFormer/img_backbone+img_neck",
            "profile": profile.to_metadata(),
            "checkpoint_key_scheme": {
                "backbone_prefix": key_map.backbone_prefix,
                "neck_prefix": key_map.neck_prefix,
                "normalization": "PyTorch BatchNorm weight/bias/running_mean/running_var",
            },
            "ignored_checkpoint_key_count": len(ignored_keys),
            "ignored_checkpoint_keys": ignored_keys,
            "framework_imported": False,
        },
    )
    if any(node.op not in _ALLOWED_NODE_OPS for node in graph.nodes):
        raise BevFormerBackboneError("backbone graph contains a non-canonical op")
    ShapeInferencePass().run(graph)
    graph.validate()
    actual = graph.value_specs[output_name]
    expected = TensorSpec(profile.output_shape, profile.parameter_dtype)
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise BevFormerBackboneError(
            "inferred FPN output does not match profile: "
            f"got {actual.shape}/{actual.dtype}, "
            f"expected {expected.shape}/{expected.dtype}"
        )
    return graph


# Concise aliases after the caller has selected the BEVFormer backbone frontend.
build_backbone_graph = build_bevformer_backbone_graph
validate_weights = validate_checkpoint_weights


__all__ = [
    "BEVFORMER_R50_FPN_PROFILE",
    "MMDET_CHECKPOINT_KEYS",
    "BevFormerBackboneError",
    "CheckpointTensorSpec",
    "MmdetCheckpointKeyMap",
    "ResNetFpnProfile",
    "build_backbone_graph",
    "build_bevformer_backbone_graph",
    "required_checkpoint_specs",
    "validate_checkpoint_weights",
    "validate_weights",
]
