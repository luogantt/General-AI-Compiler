"""UniAD's currently implemented component: one stage-2 encoder FFN.

This adapter does not represent a full tracking/planning UniAD network. Its
verified bundle contract is explicit, so future complete-model adapters can be
registered separately without changing the common compiler interface.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gac.core.ir import Graph
    from gac.models.uniad.artifact import UniADFFNBundle


class UniADFFNAdapter:
    name = "uniad-ffn"
    scope = "stage2-bev-encoder-single-layer-ffn"

    def load(self, source: str | Path) -> Graph:
        from hashlib import file_digest
        from gac.models.uniad.artifact import load_bundle

        path = Path(source)
        graph = self.build(load_bundle(path))
        with path.open("rb") as stream:
            digest = file_digest(stream, "sha256").hexdigest()
        graph.metadata.update(source_bundle_path=str(path), source_bundle_sha256=digest)
        return graph

    def build(self, data: UniADFFNBundle) -> Graph:
        from gac.core.errors import ValidationError
        from gac.models.common.ffn import TransformerFFNConfig, build_transformer_ffn_graph
        from gac.models.uniad.artifact import UniADFFNBundle
        from gac.models.uniad.contract import UniADFFNContract

        if not isinstance(data, UniADFFNBundle) or not isinstance(data.contract, UniADFFNContract):
            raise ValidationError("UniADFFNAdapter.build requires a UniADFFNBundle")
        profile = data.contract
        config = TransformerFFNConfig(
            batch=profile.batch, tokens=profile.tokens, embed_dim=profile.embed_dim, hidden_dim=profile.hidden_dim,
            post_layer_norm=True, layer_norm_epsilon=profile.layer_norm_epsilon,
        )
        prefix = f"UniAD.stage2.encoder.layers.{profile.layer}.ffn"
        graph = build_transformer_ffn_graph(
            data.weights, config, name=f"uniad_stage2_encoder_layer_{profile.layer}_ffn",
            sources={node: f"{prefix}.{stage}" for node, stage in (
                ("fc1_matmul", "fc1"), ("fc1_bias_add", "fc1"), ("relu", "relu"),
                ("fc2_matmul", "fc2"), ("fc2_bias_add", "fc2"),
                ("residual_add", "residual"), ("post_layer_norm", "norm"),
            )},
        )
        graph.metadata.update({
            "kind": "uniad-stage2-bev-encoder-ffn", "model_adapter": self.name, "model_family": "uniad",
            "model_scope": self.scope, "full_model": False, "layer": profile.layer,
            "source_weights_format": "uniad-ffn-bundle", "contract": profile.to_dict(),
        })
        return graph
