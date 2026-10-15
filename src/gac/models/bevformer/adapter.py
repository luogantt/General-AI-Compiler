"""Full BEVFormer streaming-step adapter, ending at raw detection outputs."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gac.artifacts.weights import LoadedWeights
    from gac.core.ir import Graph
    from gac.models.bevformer.step_pipeline import BevFormerStepPipelineConfig


class BEVFormerStepAdapter:
    name = "bevformer-step"
    scope = "complete-cold-and-warm-step-to-raw-detections"

    def _annotate(self, graph: Graph) -> Graph:
        graph.metadata.update(model_adapter=self.name, model_family="bevformer", model_scope=self.scope)
        return graph

    def load(self, source: str | Path, *, config: BevFormerStepPipelineConfig | None = None,
             expected_checkpoint_sha256: str | None = None) -> Graph:
        from gac.models.bevformer.frontends.graph import load_bevformer_step_graph

        return self._annotate(load_bevformer_step_graph(source, config, expected_checkpoint_sha256))

    def build(self, data: LoadedWeights, *, config: BevFormerStepPipelineConfig | None = None) -> Graph:
        from gac.artifacts.weights import LoadedWeights
        from gac.core.errors import ValidationError
        from gac.models.bevformer.frontends.graph import _weight_groups
        from gac.models.bevformer.step_pipeline import (
            BevFormerStepPipelineConfig,
            build_bevformer_step_pipeline_graph,
        )

        if not isinstance(data, LoadedWeights):
            raise ValidationError("BEVFormerStepAdapter.build requires verified LoadedWeights")
        if config is not None and not isinstance(config, BevFormerStepPipelineConfig):
            raise ValidationError("config must be a BevFormerStepPipelineConfig")
        selected = config if config is not None else BevFormerStepPipelineConfig()
        encoder, decoder = _weight_groups(data, selected)
        graph = build_bevformer_step_pipeline_graph(
            encoder, decoder, selected, name="bevformer_complete_cold_and_warm_streaming_step",
        )
        graph.metadata.update({
            "importer": "bevformer-step-data-only", "source_weights_format": "gacweights",
            "source_checkpoint_sha256": data.source_checkpoint_sha256, "loaded_weight_count": len(data),
            "component": "streaming.complete_cold_and_warm_step", "host_postprocess": "decode_nms_free",
            "postprocess": "explicit host NMS-free Top-K and box decoding after raw graph outputs",
        })
        graph.validate()
        return self._annotate(graph)
