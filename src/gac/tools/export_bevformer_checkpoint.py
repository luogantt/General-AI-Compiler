"""Export trusted BEVFormer PyTorch checkpoints to ``.gacweights``.

PyTorch is imported only after command-line validation and checkpoint hashing.
The flag name ``--trusted-checkpoint`` is intentional: old BEVFormer/PyTorch
environments deserialize pickle-backed checkpoint files, which must never be
accepted from an untrusted source.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from hashlib import sha256
import hmac
import importlib
import inspect
import json
from pathlib import Path
import re
import sys
from typing import Any

import numpy as np

from gac.core.errors import ValidationError
from gac.artifacts.weights import write_weights


DEFAULT_PREFIXES = ("img_backbone.", "img_neck.", "pts_bbox_head.")
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")


def checkpoint_sha256(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Hash a local checkpoint without importing a machine-learning framework."""

    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValidationError("chunk_size must be a positive integer")
    checkpoint = Path(path)
    if not checkpoint.is_file():
        raise ValidationError(f"trusted checkpoint is not a local file: {checkpoint}")
    digest = sha256()
    try:
        with checkpoint.open("rb") as stream:
            while True:
                block = stream.read(chunk_size)
                if not block:
                    break
                digest.update(block)
    except OSError as exc:
        raise ValidationError(f"cannot read trusted checkpoint {checkpoint}: {exc}") from exc
    return digest.hexdigest()


def _validated_expected_digest(value: str | None) -> str | None:
    if value is None:
        return None
    if _SHA256_RE.fullmatch(value) is None:
        raise ValidationError("--expected-sha256 must contain exactly 64 hexadecimal characters")
    return value.lower()


def _validated_prefixes(prefixes: Sequence[str] | None) -> tuple[str, ...]:
    selected = tuple(DEFAULT_PREFIXES if prefixes is None else prefixes)
    if not selected or any(not isinstance(prefix, str) or not prefix or "\x00" in prefix for prefix in selected):
        raise ValidationError("at least one non-empty --prefix is required")
    if len(set(selected)) != len(selected):
        raise ValidationError("duplicate --prefix values are not allowed")
    return selected


def _load_torch() -> Any:
    try:
        return importlib.import_module("torch")
    except (ImportError, OSError) as exc:
        raise ValidationError(
            "PyTorch is required only for checkpoint export. Run this command in the "
            "trusted official BEVFormer environment."
        ) from exc


def _torch_load(torch: Any, checkpoint: Path) -> Any:
    """Load on CPU, explicitly opting into pickle only after the trust flag."""

    kwargs: dict[str, Any] = {"map_location": "cpu"}
    try:
        parameters = inspect.signature(torch.load).parameters
    except (TypeError, ValueError):
        parameters = {}
    # PyTorch 2.6 defaults weights_only=True, while the BEVFormer reference
    # environment (PyTorch 1.9) has no such parameter.  Explicit trust allows
    # legacy checkpoint metadata in either version.
    if "weights_only" in parameters:
        kwargs["weights_only"] = False
    try:
        return torch.load(str(checkpoint), **kwargs)
    except Exception as exc:
        raise ValidationError(f"PyTorch could not load trusted checkpoint {checkpoint}: {exc}") from exc


def _state_dict(checkpoint: Any) -> Mapping[Any, Any]:
    if not isinstance(checkpoint, Mapping):
        raise ValidationError("checkpoint root must be a mapping")
    if "state_dict" in checkpoint:
        state = checkpoint["state_dict"]
        if not isinstance(state, Mapping):
            raise ValidationError("checkpoint['state_dict'] must be a mapping")
        return state
    return checkpoint


def _normalize_state_dict(
    torch: Any,
    state: Mapping[Any, Any],
    prefixes: Sequence[str],
) -> dict[str, np.ndarray]:
    normalized_sources: dict[str, str] = {}
    normalized_items: list[tuple[str, Any]] = []
    for original_name, value in state.items():
        if not isinstance(original_name, str) or not original_name:
            raise ValidationError("state_dict keys must be non-empty strings")
        normalized = original_name[len("module.") :] if original_name.startswith("module.") else original_name
        previous = normalized_sources.get(normalized)
        if previous is not None:
            raise ValidationError(
                f"state_dict keys {previous!r} and {original_name!r} normalize to duplicate key {normalized!r}"
            )
        normalized_sources[normalized] = original_name
        normalized_items.append((normalized, value))

    result: dict[str, np.ndarray] = {}
    for name, value in normalized_items:
        if not any(name.startswith(prefix) for prefix in prefixes):
            continue
        try:
            is_tensor = bool(torch.is_tensor(value))
        except Exception as exc:
            raise ValidationError(f"cannot inspect selected state value {name!r}: {exc}") from exc
        if not is_tensor:
            raise ValidationError(f"selected state value {name!r} is not a torch Tensor")
        if bool(getattr(value, "is_sparse", False)):
            raise ValidationError(f"selected tensor {name!r} is sparse")
        strided = getattr(torch, "strided", None)
        if strided is not None and getattr(value, "layout", strided) != strided:
            raise ValidationError(f"selected tensor {name!r} does not use strided layout")
        if bool(getattr(value, "is_quantized", False)):
            raise ValidationError(f"selected tensor {name!r} is quantized")
        try:
            tensor = value.detach().cpu().contiguous()
            device = getattr(tensor, "device", None)
            if device is not None and getattr(device, "type", "cpu") != "cpu":
                raise ValidationError(f"selected tensor {name!r} could not be moved to CPU")
            array = tensor.numpy()
        except ValidationError:
            raise
        except Exception as exc:
            raise ValidationError(f"cannot convert selected tensor {name!r} to NumPy: {exc}") from exc
        if not isinstance(array, np.ndarray) or array.dtype.hasobject:
            raise ValidationError(f"selected tensor {name!r} did not produce a safe NumPy array")
        if array.dtype.kind not in "biufc" or array.dtype.fields is not None:
            raise ValidationError(f"selected tensor {name!r} has unsupported dtype {array.dtype}")
        if array.dtype.kind in "fc" and not np.isfinite(array).all():
            raise ValidationError(f"selected tensor {name!r} contains non-finite values")
        result[name] = np.ascontiguousarray(array)
    if not result:
        raise ValidationError(f"checkpoint contains no tensors matching prefixes {tuple(prefixes)!r}")
    return result


def export_bevformer_checkpoint(
    checkpoint: str | Path,
    output: str | Path,
    *,
    expected_sha256: str | None = None,
    prefixes: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Hash, explicitly deserialize, filter, and export one trusted checkpoint."""

    checkpoint_path = Path(checkpoint)
    actual_digest = checkpoint_sha256(checkpoint_path)
    expected_digest = _validated_expected_digest(expected_sha256)
    if expected_digest is not None and not hmac.compare_digest(actual_digest, expected_digest):
        raise ValidationError(
            f"checkpoint SHA256 mismatch: expected {expected_digest}, got {actual_digest}"
        )
    selected_prefixes = _validated_prefixes(prefixes)
    torch = _load_torch()
    checkpoint_data = _torch_load(torch, checkpoint_path)
    arrays = _normalize_state_dict(torch, _state_dict(checkpoint_data), selected_prefixes)
    destination = write_weights(
        output,
        arrays,
        source_checkpoint_sha256=actual_digest,
    )
    return {
        "format": "gacweights",
        "output": str(destination),
        "source_checkpoint_sha256": actual_digest,
        "prefixes": list(selected_prefixes),
        "weight_count": len(arrays),
        "tensor_bytes": sum(array.nbytes for array in arrays.values()),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export selected tensors from an explicitly trusted BEVFormer checkpoint."
    )
    parser.add_argument(
        "--trusted-checkpoint",
        required=True,
        type=Path,
        help="local checkpoint to deserialize; only use a file whose origin you trust",
    )
    parser.add_argument("-o", "--output", required=True, type=Path, help="output .gacweights path")
    parser.add_argument(
        "--expected-sha256",
        help="optional expected checkpoint SHA256; export stops before importing torch on mismatch",
    )
    parser.add_argument(
        "--prefix",
        dest="prefixes",
        action="append",
        help=(
            "state_dict prefix to export; repeat as needed "
            "(default: img_backbone., img_neck., pts_bbox_head.)"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    try:
        report = export_bevformer_checkpoint(
            arguments.trusted_checkpoint,
            arguments.output,
            expected_sha256=arguments.expected_sha256,
            prefixes=arguments.prefixes,
        )
    except ValidationError as exc:
        parser.error(str(exc))
    json.dump(report, sys.stdout, ensure_ascii=False, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
