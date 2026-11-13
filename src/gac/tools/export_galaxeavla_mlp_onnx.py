"""Export the fixed Galaxea G0.5 Action Expert MLP slice to ONNX.

The input is a verified, data-only ``.gacweights`` archive.  ONNX remains an
optional dependency and is imported only after the weights, trusted-extractor
attestation and static model contract have been validated.  The checkpoint
digest stored in the archive is an attestation, not a cryptographic binding
between the original checkpoint and the extracted tensors.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from hashlib import sha256
import importlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

from gac.core.errors import ValidationError


ONNX_OPSET = 17
_EXPECTED_NODE_OPS = ("MatMul", "Sigmoid", "Mul", "MatMul", "Mul", "MatMul")


def _load_onnx() -> Any:
    try:
        return importlib.import_module("onnx")
    except (ImportError, OSError) as exc:
        raise ValidationError(
            "ONNX export requires the optional 'onnx' dependency; install it with "
            "`pip install 'gac-heterogeneous[onnx]'`"
        ) from exc


def _atomic_write(path: Path, payload: bytes) -> None:
    destination = path.resolve()
    temporary_name: str | None = None
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{destination.name}-",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as temporary:
            temporary_name = temporary.name
            temporary.write(payload)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_name, destination)
        temporary_name = None
    except OSError as exc:
        raise ValidationError(f"cannot atomically write ONNX model {destination}: {exc}") from exc
    finally:
        if temporary_name is not None:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass


def _file_sha256(path: Path) -> str:
    digest = sha256()
    try:
        with path.open("rb") as stream:
            while block := stream.read(1024 * 1024):
                digest.update(block)
    except OSError as exc:
        raise ValidationError(f"cannot hash source weights {path}: {exc}") from exc
    return digest.hexdigest()


def _value_info(onnx: Any, name: str, shape: tuple[int, ...]) -> Any:
    return onnx.helper.make_tensor_value_info(
        name, onnx.TensorProto.FLOAT, list(shape)
    )


def _onnx_model(graph: Any, onnx: Any) -> Any:
    node_ops = tuple(node.op for node in graph.nodes)
    if node_ops != _EXPECTED_NODE_OPS:
        raise ValidationError(
            f"GalaxeaVLA MLP graph requires node sequence {_EXPECTED_NODE_OPS}, got {node_ops}"
        )
    if any(node.attrs for node in graph.nodes):
        raise ValidationError("GalaxeaVLA MLP ONNX nodes must not carry target-specific attributes")

    initializers = [
        onnx.numpy_helper.from_array(graph.constants[name], name=name)
        for name in ("gate_proj_weight", "up_proj_weight", "down_proj_weight")
    ]
    nodes = [
        onnx.helper.make_node(
            node.op,
            list(node.inputs),
            list(node.outputs),
            name=node.id,
        )
        for node in graph.nodes
    ]
    input_infos = [
        _value_info(onnx, name, tuple(int(dim) for dim in spec.shape))
        for name, spec in graph.inputs.items()
    ]
    output_infos = [
        _value_info(
            onnx,
            name,
            tuple(int(dim) for dim in graph.value_specs[name].shape),
        )
        for name in graph.outputs
    ]
    onnx_graph = onnx.helper.make_graph(
        nodes,
        graph.name,
        input_infos,
        output_infos,
        initializer=initializers,
        doc_string=(
            "Galaxea G0.5 Qwen3.5 Action Expert layer-0 SwiGLU MLP; "
            "this is not the full VLA model"
        ),
    )
    model = onnx.helper.make_model(
        onnx_graph,
        producer_name="General-AI-Compiler",
        opset_imports=[onnx.helper.make_operatorsetid("", ONNX_OPSET)],
        ir_version=8,
    )
    onnx.helper.set_model_props(
        model,
        {
            "gac.model_adapter": graph.metadata["model_adapter"],
            "gac.model_scope": graph.metadata["model_scope"],
            "gac.upstream_repository": graph.metadata["upstream_repository"],
            "gac.upstream_source_revision": graph.metadata[
                "upstream_source_revision"
            ],
            "gac.upstream_license_url": graph.metadata["upstream_license_url"],
            "gac.upstream_usage_notice": graph.metadata["upstream_usage_notice"],
            "gac.full_model": "false",
            "gac.official_g05_dimensions": str(
                bool(graph.metadata["official_g05_dimensions"])
            ).lower(),
            "gac.source_checkpoint_sha256": graph.metadata[
                "source_checkpoint_sha256"
            ],
            "gac.source_checkpoint_digest_kind": graph.metadata[
                "source_checkpoint_digest_kind"
            ],
            "gac.expected_checkpoint_sha256_matched": str(
                bool(graph.metadata["expected_checkpoint_sha256_matched"])
            ).lower(),
            "gac.source_weights_sha256": graph.metadata["source_weights_sha256"],
            "gac.weight_layout": graph.metadata["weight_layout"],
            "gac.contract": json.dumps(
                graph.metadata["contract"],
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            ),
        },
    )
    try:
        onnx.checker.check_model(model)
    except Exception as exc:
        raise ValidationError(f"generated ONNX model failed validation: {exc}") from exc
    return model


def _serialize_model(model: Any) -> bytes:
    try:
        return model.SerializeToString(deterministic=True)
    except TypeError:
        return model.SerializeToString()


def export_galaxeavla_mlp_onnx(
    weights: str | Path,
    output: str | Path,
    *,
    expected_checkpoint_sha256: str | None = None,
    batch: int = 1,
    tokens: int = 32,
    hidden_size: int = 1024,
    intermediate_size: int = 4096,
) -> dict[str, Any]:
    """Validate a data-only slice and atomically publish one static ONNX model."""

    from gac.models.api import get_model_adapter
    from gac.models.galaxeavla.contract import GalaxeaQwen35MLPContract

    if expected_checkpoint_sha256 is None:
        raise ValidationError(
            "expected_checkpoint_sha256 is required for publishable ONNX output; "
            "it checks the trusted extractor's checkpoint-digest attestation"
        )

    source = Path(weights).resolve()
    destination = Path(output).resolve()
    if source == destination:
        raise ValidationError("weights and ONNX output must resolve to different paths")
    source_weights_sha256_before = _file_sha256(source)
    contract = GalaxeaQwen35MLPContract(
        batch=batch,
        tokens=tokens,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
    )
    graph = get_model_adapter("galaxeavla-qwen35-mlp").load(
        source,
        contract=contract,
        expected_checkpoint_sha256=expected_checkpoint_sha256,
    )
    source_weights_sha256_after = _file_sha256(source)
    if source_weights_sha256_before != source_weights_sha256_after:
        raise ValidationError("source .gacweights changed while it was being loaded")
    graph.metadata["source_weights_sha256"] = source_weights_sha256_after

    onnx = _load_onnx()
    model = _onnx_model(graph, onnx)
    payload = _serialize_model(model)
    digest = sha256(payload).hexdigest()
    _atomic_write(destination, payload)

    node_names = [node.id for node in graph.nodes]
    node_ops = [node.op for node in graph.nodes]
    initializer_shapes = {
        name: list(graph.constants[name].shape)
        for name in ("gate_proj_weight", "up_proj_weight", "down_proj_weight")
    }
    return {
        "format": "onnx",
        "opset": ONNX_OPSET,
        "output": str(destination),
        "onnx_sha256": digest,
        "size_bytes": len(payload),
        "source_checkpoint_sha256": graph.metadata["source_checkpoint_sha256"],
        "source_weights_sha256": source_weights_sha256_after,
        "model_adapter": graph.metadata["model_adapter"],
        "model_scope": graph.metadata["model_scope"],
        "upstream_repository": graph.metadata["upstream_repository"],
        "upstream_source_revision": graph.metadata["upstream_source_revision"],
        "upstream_license_url": graph.metadata["upstream_license_url"],
        "upstream_usage_notice": graph.metadata["upstream_usage_notice"],
        "full_model": False,
        "official_g05_dimensions": contract.official_g05_dimensions,
        "weight_layout": graph.metadata["weight_layout"],
        "source_checkpoint_digest_kind": graph.metadata[
            "source_checkpoint_digest_kind"
        ],
        "expected_checkpoint_sha256_matched": graph.metadata[
            "expected_checkpoint_sha256_matched"
        ],
        "input": {
            "name": "hidden_states",
            "shape": list(contract.input_shape),
            "dtype": contract.dtype,
        },
        "output_tensor": {
            "name": "output",
            "shape": list(contract.output_shape),
            "dtype": contract.dtype,
        },
        "initializers": initializer_shapes,
        "nodes": {
            "count": len(node_ops),
            "names": node_names,
            "op_types": node_ops,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("weights", type=Path, help="verified data-only .gacweights archive")
    parser.add_argument("-o", "--output", required=True, type=Path, help="atomic ONNX output path")
    parser.add_argument(
        "--expected-checkpoint-sha256",
        required=True,
        help=(
            "required expected source-checkpoint SHA256; compares the trusted "
            "extractor attestation recorded by the .gacweights manifest"
        ),
    )
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--hidden-size", type=int, default=1024)
    parser.add_argument("--intermediate-size", type=int, default=4096)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    try:
        summary = export_galaxeavla_mlp_onnx(
            arguments.weights,
            arguments.output,
            expected_checkpoint_sha256=arguments.expected_checkpoint_sha256,
            batch=arguments.batch,
            tokens=arguments.tokens,
            hidden_size=arguments.hidden_size,
            intermediate_size=arguments.intermediate_size,
        )
    except (ValidationError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(summary, allow_nan=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ONNX_OPSET",
    "build_parser",
    "export_galaxeavla_mlp_onnx",
    "main",
]
