#!/usr/bin/env python3
"""Export one UniAD 2.0 BEV encoder FFN into the compiler interchange bundle.

Run this command in the official UniAD Python 3.9 / Torch 2.0 environment.
The output contains canonical NumPy weights and a checksummed JSON contract;
it deliberately does not import the Python 3.11 General AI Compiler runtime.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


from gac.models.uniad import (
    UniADFFNContract,
    encoder_ffn_checkpoint_keys,
    extract_encoder_ffn_weights,
    write_bundle,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _unwrap_state_dict(checkpoint: Any) -> Mapping[str, Any]:
    if not isinstance(checkpoint, Mapping):
        raise TypeError("checkpoint root must be a mapping")
    for name in ("state_dict", "model"):
        candidate = checkpoint.get(name)
        if isinstance(candidate, Mapping):
            return candidate
    return checkpoint


def _selected_numpy_state_dict(state_dict: Mapping[str, Any], layer: int) -> dict[str, np.ndarray]:
    required = set(encoder_ffn_checkpoint_keys(layer).values())
    selected: dict[str, np.ndarray] = {}
    normalized_sources: dict[str, str] = {}
    for raw_name, value in state_dict.items():
        if not isinstance(raw_name, str):
            continue
        name = raw_name[7:] if raw_name.startswith("module.") else raw_name
        previous = normalized_sources.get(name)
        if previous is not None:
            raise ValueError(
                f"checkpoint keys {previous!r} and {raw_name!r} normalize to "
                f"duplicate key {name!r}"
            )
        normalized_sources[name] = raw_name
        if name not in required:
            continue
        if not hasattr(value, "detach"):
            raise TypeError(f"checkpoint value {raw_name!r} is not a Torch tensor")
        selected[name] = value.detach().cpu().numpy()
    return selected


def _torch_golden(
    query: np.ndarray,
    weights: Mapping[str, np.ndarray],
    epsilon: float,
) -> np.ndarray:
    import torch
    import torch.nn.functional as functional

    with torch.no_grad():
        value = torch.from_numpy(query)
        residual = value
        value = functional.linear(
            value,
            torch.from_numpy(weights["fc1_weight"].T.copy()),
            torch.from_numpy(weights["fc1_bias"]),
        )
        value = functional.relu(value)
        value = functional.linear(
            value,
            torch.from_numpy(weights["fc2_weight"].T.copy()),
            torch.from_numpy(weights["fc2_bias"]),
        )
        value = functional.layer_norm(
            residual + value,
            (query.shape[-1],),
            torch.from_numpy(weights["norm_weight"]),
            torch.from_numpy(weights["norm_bias"]),
            epsilon,
        )
    return np.ascontiguousarray(value.numpy(), dtype=np.float32)


def _torch_load(torch: Any, path: Path, *, trusted_checkpoint: bool) -> Any:
    """Use the data-only loader when available and gate legacy pickle explicitly."""

    try:
        parameters = inspect.signature(torch.load).parameters
    except (TypeError, ValueError):
        parameters = {}
    if "weights_only" in parameters:
        return torch.load(str(path), map_location="cpu", weights_only=True)
    if not trusted_checkpoint:
        raise RuntimeError(
            "this Torch version cannot load weights without pickle; pass "
            "--trusted-checkpoint only for a checkpoint from a trusted source"
        )
    return torch.load(str(path), map_location="cpu")


def _load_checkpoint(path: Path, *, trusted_checkpoint: bool) -> Any:
    import torch

    return _torch_load(torch, path, trusted_checkpoint=trusted_checkpoint)


def _resolve_export_paths(checkpoint: Path, output: Path) -> tuple[Path, Path, Path]:
    """Resolve the three output-boundary paths and reject every alias."""

    checkpoint_path = checkpoint.resolve()
    bundle_path = output.resolve()
    provenance_path = Path(str(bundle_path) + ".provenance.json").resolve()
    named_paths = {
        "checkpoint": checkpoint_path,
        "output": bundle_path,
        "provenance": provenance_path,
    }
    names = tuple(named_paths)
    for index, left_name in enumerate(names):
        for right_name in names[index + 1 :]:
            if named_paths[left_name] == named_paths[right_name]:
                raise ValueError(
                    "{} and {} must not resolve to the same path: {}".format(
                        left_name, right_name, named_paths[left_name]
                    )
                )
    return checkpoint_path, bundle_path, provenance_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--layer", type=int, default=0, choices=range(6))
    parser.add_argument(
        "--trusted-checkpoint",
        action="store_true",
        help=(
            "allow a legacy Torch version without weights_only support to "
            "deserialize this checkpoint with pickle"
        ),
    )
    parser.add_argument(
        "--with-synthetic-golden",
        action="store_true",
        help=(
            "also export a deterministic [1,40000,256] query and Torch eager "
            "golden output (about 82 MiB before NPZ overhead)"
        ),
    )
    parser.add_argument("--seed", type=int, default=20261011)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    try:
        checkpoint_path, bundle_path, provenance_path = _resolve_export_paths(
            args.checkpoint, args.output
        )
    except ValueError as exc:
        parser.error(str(exc))
    if not checkpoint_path.is_file():
        parser.error(f"checkpoint does not exist: {checkpoint_path}")

    contract = UniADFFNContract(layer=args.layer)
    checkpoint = _load_checkpoint(
        checkpoint_path, trusted_checkpoint=args.trusted_checkpoint
    )
    state_dict = _selected_numpy_state_dict(
        _unwrap_state_dict(checkpoint), args.layer
    )
    weights = extract_encoder_ffn_weights(state_dict, args.layer)

    query = None
    golden = None
    if args.with_synthetic_golden:
        generator = np.random.RandomState(args.seed)
        query = np.ascontiguousarray(
            generator.normal(0.0, 0.1, contract.input_shape).astype(np.float32)
        )
        golden = _torch_golden(query, weights, contract.layer_norm_epsilon)

    write_bundle(
        bundle_path,
        contract,
        weights,
        sample=(
            {"query": query, "output": golden}
            if query is not None and golden is not None
            else None
        ),
    )
    provenance = {
        "schema_version": 1,
        "source": "UniAD 2.0 PyTorch checkpoint",
        "bundle_file": bundle_path.name,
        "bundle_sha256": _sha256(bundle_path),
        "checkpoint_file": checkpoint_path.name,
        "checkpoint_sha256": _sha256(checkpoint_path),
        "checkpoint_key_prefix": (
            "pts_bbox_head.transformer.encoder.layers.%d" % args.layer
        ),
        "golden_runtime": (
            "torch-eager-cpu" if args.with_synthetic_golden else None
        ),
        "synthetic_seed": args.seed if args.with_synthetic_golden else None,
    }
    provenance_path.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(bundle_path)
    print("bundle_sha256=" + provenance["bundle_sha256"])
    print("provenance=" + str(provenance_path))


if __name__ == "__main__":
    main()
