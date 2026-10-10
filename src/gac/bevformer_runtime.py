"""Stateful BEVFormer step wrapper with an explicit, transactional ABI.

The compiled graph remains a pure function.  This wrapper owns the previous
BEV and vehicle pose between calls, converts absolute CAN-bus values into the
delta convention used by the reference model, and commits state only after a
successful and fully validated invocation.
"""

from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import Any, Mapping, Protocol, runtime_checkable

import numpy as np

from .errors import BackendError, ValidationError


@runtime_checkable
class StepSession(Protocol):
    """Minimal protocol implemented by CUDA and test sessions."""

    def run(self, inputs: Mapping[str, np.ndarray], **kwargs: Any) -> Any: ...


@dataclass(frozen=True, slots=True)
class BevFormerStepContract:
    batch: int = 1
    cameras: int = 6
    channels: int = 3
    image_h: int = 480
    image_w: int = 800
    bev_queries: int = 2500
    embed_dim: int = 256
    object_queries: int = 900
    classes: int = 10
    bbox_dims: int = 10
    can_bus_dims: int = 18
    image_dtype: str = "float16"
    state_dtype: str = "float16"

    def __post_init__(self) -> None:
        for name in (
            "batch", "cameras", "channels", "image_h", "image_w",
            "bev_queries", "embed_dim", "object_queries", "classes",
            "bbox_dims", "can_bus_dims",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValidationError(f"BEVFormer contract {name} must be a positive integer")
        for name in ("image_dtype", "state_dtype"):
            try:
                dtype = np.dtype(getattr(self, name))
            except TypeError as exc:
                raise ValidationError(f"BEVFormer contract {name} is invalid") from exc
            if not np.issubdtype(dtype, np.floating):
                raise ValidationError(f"BEVFormer contract {name} must be floating point")
            object.__setattr__(self, name, dtype.name)
        if self.batch != 1:
            raise ValidationError("the first BEVFormer runtime profile requires batch=1")
        if self.can_bus_dims < 4:
            raise ValidationError("can_bus_dims must include xyz position and heading")

    @property
    def image_shape(self) -> tuple[int, ...]:
        return (self.batch, self.cameras, self.channels, self.image_h, self.image_w)

    @property
    def lidar2img_shape(self) -> tuple[int, ...]:
        return (self.batch, self.cameras, 4, 4)

    @property
    def can_bus_shape(self) -> tuple[int, ...]:
        return (self.batch, self.can_bus_dims)

    @property
    def prev_bev_shape(self) -> tuple[int, ...]:
        return (self.batch, self.bev_queries, self.embed_dim)

    @property
    def cls_shape(self) -> tuple[int, ...]:
        return (self.batch, self.object_queries, self.classes)

    @property
    def bbox_shape(self) -> tuple[int, ...]:
        return (self.batch, self.object_queries, self.bbox_dims)

    @classmethod
    def from_manifest(cls, manifest: Any) -> "BevFormerStepContract":
        """Derive the runtime contract from a validated step manifest.

        The dependency is intentionally structural so the runtime does not
        need to import the configuration reader or execute model source.
        """

        validate = getattr(manifest, "validate", None)
        if not callable(validate):
            raise ValidationError("manifest must provide validate()")
        try:
            validate()
            configuration = manifest.configuration
            inputs = {item.name: item for item in manifest.inputs}
            outputs = {item.name: item for item in manifest.outputs}
            contract = cls(
                batch=configuration.batch,
                cameras=configuration.cameras,
                channels=configuration.image_channels,
                image_h=configuration.image_input_hw[0],
                image_w=configuration.image_input_hw[1],
                bev_queries=configuration.bev_queries,
                embed_dim=configuration.embed_dim,
                object_queries=configuration.object_queries,
                classes=configuration.num_classes,
                bbox_dims=configuration.bbox_code_size,
                can_bus_dims=configuration.can_bus_dim,
                image_dtype=inputs["images"].dtype,
                state_dtype=inputs["prev_bev"].dtype,
            )
        except ValidationError:
            raise
        except Exception as exc:
            raise ValidationError(f"invalid BEVFormer step manifest: {exc}") from exc

        expected = {
            "images": (contract.image_shape, contract.image_dtype),
            "lidar2img": (contract.lidar2img_shape, "float32"),
            "can_bus_delta": (contract.can_bus_shape, "float32"),
            "prev_bev": (contract.prev_bev_shape, contract.state_dtype),
            "has_prev_bev": ((contract.batch,), "int32"),
            "cls_logits": (contract.cls_shape, "float32"),
            "bbox_regression": (contract.bbox_shape, "float32"),
            "next_bev": (contract.prev_bev_shape, contract.state_dtype),
        }
        records = {**inputs, **outputs}
        if set(records) != set(expected):
            raise ValidationError("BEVFormer manifest tensor names do not match the runtime ABI")
        for name, (shape, dtype) in expected.items():
            record = records[name]
            if tuple(record.shape) != shape or np.dtype(record.dtype).name != dtype:
                raise ValidationError(
                    f"BEVFormer manifest tensor {name!r} does not match the runtime contract"
                )
        return contract


@dataclass(frozen=True, slots=True)
class BevFormerState:
    prev_bev: np.ndarray
    scene_token: str | None
    last_position: np.ndarray
    last_angle: float
    valid: bool
    frame_index: int

    def snapshot(self) -> "BevFormerState":
        return BevFormerState(
            prev_bev=self.prev_bev.copy(),
            scene_token=self.scene_token,
            last_position=self.last_position.copy(),
            last_angle=float(self.last_angle),
            valid=bool(self.valid),
            frame_index=int(self.frame_index),
        )


@dataclass(frozen=True, slots=True)
class BevFormerStepResult:
    cls_logits: np.ndarray
    bbox_regression: np.ndarray
    next_bev: np.ndarray
    cold_start: bool
    scene_token: str
    frame_index: int
    runtime_result: Any = None


def _as_tensor(
    name: str,
    value: Any,
    *,
    shape: tuple[int, ...],
    dtype: str,
) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != shape:
        raise ValidationError(f"{name} requires shape {shape}, got {array.shape}")
    if array.dtype.name != dtype:
        raise ValidationError(f"{name} requires dtype {dtype}, got {array.dtype.name}")
    if not array.flags.c_contiguous:
        raise ValidationError(f"{name} must be C-contiguous")
    if not np.all(np.isfinite(array)):
        raise ValidationError(f"{name} must contain finite values")
    return array


class BevFormerRuntime:
    """Own BEV temporal state around a pure compiled ``bevformer_step`` graph."""

    def __init__(
        self,
        session: StepSession,
        *,
        contract: BevFormerStepContract | None = None,
    ) -> None:
        if not isinstance(session, StepSession):
            raise ValidationError("session must provide run(inputs)")
        self._session = session
        self.contract = contract or BevFormerStepContract()
        self._lock = Lock()
        self._closed = False
        self._state = self._cold_state(None)

    def _cold_state(self, scene_token: str | None) -> BevFormerState:
        return BevFormerState(
            prev_bev=np.zeros(
                self.contract.prev_bev_shape, dtype=self.contract.state_dtype
            ),
            scene_token=scene_token,
            last_position=np.zeros((3,), dtype=np.float32),
            last_angle=0.0,
            valid=False,
            frame_index=0,
        )

    @property
    def state(self) -> BevFormerState:
        with self._lock:
            return self._state.snapshot()

    def reset(self, scene_token: str | None = None) -> None:
        if scene_token is not None and (not isinstance(scene_token, str) or not scene_token):
            raise ValidationError("scene_token must be a non-empty string or None")
        with self._lock:
            if self._closed:
                raise BackendError("BEVFormer runtime is closed")
            self._state = self._cold_state(scene_token)

    @staticmethod
    def _outputs(runtime_result: Any) -> Mapping[str, Any]:
        outputs = getattr(runtime_result, "outputs", runtime_result)
        if not isinstance(outputs, Mapping):
            raise BackendError("BEVFormer session must return a mapping or an object with .outputs")
        required = {"cls_logits", "bbox_regression", "next_bev"}
        if set(outputs) != required:
            raise BackendError(
                f"BEVFormer session outputs must be {sorted(required)}, got {sorted(outputs)}"
            )
        return outputs

    def step(
        self,
        images: np.ndarray,
        lidar2img: np.ndarray,
        can_bus: np.ndarray,
        scene_token: str,
    ) -> BevFormerStepResult:
        if not isinstance(scene_token, str) or not scene_token:
            raise ValidationError("scene_token must be a non-empty string")
        if not self._lock.acquire(blocking=False):
            raise BackendError("concurrent BEVFormerRuntime calls are not supported")
        try:
            if self._closed:
                raise BackendError("BEVFormer runtime is closed")
            contract = self.contract
            images = _as_tensor(
                "images", images, shape=contract.image_shape, dtype=contract.image_dtype
            )
            lidar2img = _as_tensor(
                "lidar2img", lidar2img,
                shape=contract.lidar2img_shape, dtype="float32",
            )
            absolute_can_bus = _as_tensor(
                "can_bus", can_bus, shape=contract.can_bus_shape, dtype="float32"
            )

            previous = self._state
            same_scene = previous.valid and previous.scene_token == scene_token
            cold_start = not same_scene
            base = previous if same_scene else self._cold_state(scene_token)
            delta = absolute_can_bus.copy()
            if same_scene:
                delta[0, :3] -= previous.last_position
                delta[0, -1] -= np.float32(previous.last_angle)
            else:
                # Match the official BEVFormer reset path: translation and
                # frame-to-frame rotation have no predecessor, while the
                # quaternion, absolute heading, and other CAN-bus fields stay
                # available to the learned CAN-bus MLP.
                delta[0, :3] = np.float32(0.0)
                delta[0, -1] = np.float32(0.0)

            inputs = {
                "images": images,
                "lidar2img": lidar2img,
                "can_bus_delta": np.ascontiguousarray(delta),
                "prev_bev": np.ascontiguousarray(base.prev_bev),
                "has_prev_bev": np.asarray([0 if cold_start else 1], dtype=np.int32),
            }
            runtime_result = self._session.run(inputs)
            outputs = self._outputs(runtime_result)
            cls_logits = _as_tensor(
                "cls_logits", outputs["cls_logits"],
                shape=contract.cls_shape, dtype="float32",
            )
            bbox_regression = _as_tensor(
                "bbox_regression", outputs["bbox_regression"],
                shape=contract.bbox_shape, dtype="float32",
            )
            next_bev = _as_tensor(
                "next_bev", outputs["next_bev"],
                shape=contract.prev_bev_shape, dtype=contract.state_dtype,
            )

            frame_index = base.frame_index + 1
            self._state = BevFormerState(
                prev_bev=next_bev.copy(),
                scene_token=scene_token,
                last_position=absolute_can_bus[0, :3].copy(),
                last_angle=float(absolute_can_bus[0, -1]),
                valid=True,
                frame_index=frame_index,
            )
            return BevFormerStepResult(
                cls_logits=cls_logits.copy(),
                bbox_regression=bbox_regression.copy(),
                next_bev=next_bev.copy(),
                cold_start=cold_start,
                scene_token=scene_token,
                frame_index=frame_index,
                runtime_result=runtime_result,
            )
        finally:
            self._lock.release()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            close = getattr(self._session, "close", None)
            if callable(close):
                close()
            self._closed = True

    def __enter__(self) -> "BevFormerRuntime":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


__all__ = [
    "BevFormerRuntime",
    "BevFormerState",
    "BevFormerStepContract",
    "BevFormerStepResult",
    "StepSession",
]
