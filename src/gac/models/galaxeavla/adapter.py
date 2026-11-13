"""Lazy model adapter for Galaxea G0.5's first compiler slice."""

from __future__ import annotations

from pathlib import Path
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gac.artifacts.weights import LoadedWeights
    from gac.core.ir import Graph
    from gac.models.galaxeavla.contract import GalaxeaQwen35MLPContract


class GalaxeaVLAQwen35MLPAdapter:
    """Build Action Expert layer-zero's MLP from verified data-only weights."""

    name = "galaxeavla-qwen35-mlp"
    scope = "action-expert-layer0-qwen35-swiglu-mlp"

    def load(
        self,
        source: str | Path,
        *,
        contract: GalaxeaQwen35MLPContract | None = None,
        expected_checkpoint_sha256: str | None = None,
    ) -> Graph:
        from gac.artifacts.weights import load_weights

        path = Path(source)
        graph = self.build(
            load_weights(path),
            contract=contract,
            expected_checkpoint_sha256=expected_checkpoint_sha256,
        )
        graph.metadata["source_weights_path"] = str(path)
        return graph

    def build(
        self,
        data: LoadedWeights,
        *,
        contract: GalaxeaQwen35MLPContract | None = None,
        expected_checkpoint_sha256: str | None = None,
    ) -> Graph:
        from gac.artifacts.weights import LoadedWeights
        from gac.core.errors import ValidationError
        from gac.models.galaxeavla.contract import GalaxeaQwen35MLPContract, SOURCE_WEIGHT_KEYS
        from gac.models.galaxeavla.graph import (
            build_qwen35_mlp_graph,
            canonicalize_qwen35_mlp_weights,
        )

        if not isinstance(data, LoadedWeights):
            raise ValidationError(
                "GalaxeaVLAQwen35MLPAdapter.build requires verified LoadedWeights"
            )
        if expected_checkpoint_sha256 is not None:
            if (
                not isinstance(expected_checkpoint_sha256, str)
                or re.fullmatch(r"[0-9a-fA-F]{64}", expected_checkpoint_sha256) is None
            ):
                raise ValidationError(
                    "expected_checkpoint_sha256 must be a 64-character hexadecimal SHA256"
                )
            if data.source_checkpoint_sha256 != expected_checkpoint_sha256.lower():
                raise ValidationError(
                    "verified weights source checkpoint SHA256 does not match the expected digest"
                )
        selected = GalaxeaQwen35MLPContract() if contract is None else contract
        if not isinstance(selected, GalaxeaQwen35MLPContract):
            raise ValidationError("contract must be a GalaxeaQwen35MLPContract")
        canonical = canonicalize_qwen35_mlp_weights(data, selected)
        graph = build_qwen35_mlp_graph(canonical, selected)
        graph.metadata.update(
            {
                "importer": "galaxeavla-qwen35-mlp-data-only",
                "source_weights_format": "gacweights",
                "source_checkpoint_sha256": data.source_checkpoint_sha256,
                "source_checkpoint_digest_kind": (
                    "trusted-extractor-attestation-not-cryptographically-bound"
                ),
                "expected_checkpoint_sha256_matched": (
                    expected_checkpoint_sha256 is not None
                ),
                "loaded_weight_count": len(data),
                "selected_weight_count": len(SOURCE_WEIGHT_KEYS),
                "selected_weight_keys": list(SOURCE_WEIGHT_KEYS),
                "full_model": False,
            }
        )
        graph.validate()
        return graph


__all__ = ["GalaxeaVLAQwen35MLPAdapter"]
