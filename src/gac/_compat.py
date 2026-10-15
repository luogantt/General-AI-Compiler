"""Lazy compatibility imports for the pre-layered public module paths.

Canonical implementation modules are imported only when a legacy name is used.
Both names refer to the *same module object*, preserving class identity, patches
and serialized API references without dozens of forwarding source files.
"""
from __future__ import annotations

from importlib import import_module
from importlib.abc import Loader, MetaPathFinder
from importlib.util import find_spec, spec_from_loader
import sys

LEGACY_MODULE_ALIASES = {'gac.backends.cuda': 'gac.targets.cuda.compiler',
 'gac.backends.mlir': 'gac.targets.mlir',
 'gac.backends.mlir_cublas_ptx': 'gac.targets.cuda.cublas_ptx',
 'gac.backends.mlir_cuda': 'gac.targets.cuda.mlir',
 'gac.backends.mlir_ptx': 'gac.targets.cuda.ptx',
 'gac.backends.vortex': 'gac.targets.vortex.ffn',
 'gac.backends.vortex_buffered_mlir': 'gac.targets.vortex.buffered_mlir',
 'gac.backends.vortex_device_mlir': 'gac.targets.vortex.device_mlir',
 'gac.backends.vortex_graph': 'gac.targets.vortex.compiler',
 'gac.backends.vortex_graph_checks': 'gac.targets.vortex.checks',
 'gac.backends.vortex_graph_runtime': 'gac.targets.vortex.host_runtime',
 'gac.backends.vortex_graph_validation': 'gac.targets.vortex.validation',
 'gac.backends.vortex_scalar_ir': 'gac.targets.vortex.scalar_ir',
 'gac.backends.vortex_scalar_ops': 'gac.targets.vortex.scalar_ops',
 'gac.backends.vortex_spatial_ops': 'gac.targets.vortex.spatial_ops',
 'gac.bevformer': 'gac.models.bevformer.model',
 'gac.bevformer_cold_pipeline': 'gac.models.bevformer.cold_pipeline',
 'gac.bevformer_compiler': 'gac.workflows.bevformer',
 'gac.bevformer_decoder': 'gac.models.bevformer.decoder',
 'gac.bevformer_decoder_layer': 'gac.models.bevformer.decoder_layer',
 'gac.bevformer_decoder_pipeline': 'gac.models.bevformer.decoder_pipeline',
 'gac.bevformer_decoder_self': 'gac.models.bevformer.decoder_self',
 'gac.bevformer_decoder_stack': 'gac.models.bevformer.decoder_stack',
 'gac.bevformer_detection_head': 'gac.models.bevformer.detection_head',
 'gac.bevformer_detection_stack': 'gac.models.bevformer.detection_stack',
 'gac.bevformer_encoder': 'gac.models.bevformer.encoder',
 'gac.bevformer_encoder_inputs': 'gac.models.bevformer.encoder_inputs',
 'gac.bevformer_encoder_layer': 'gac.models.bevformer.encoder_layer',
 'gac.bevformer_encoder_pipeline': 'gac.models.bevformer.encoder_pipeline',
 'gac.bevformer_encoder_stack': 'gac.models.bevformer.encoder_stack',
 'gac.bevformer_feature_prep': 'gac.models.bevformer.feature_prep',
 'gac.bevformer_ffn': 'gac.models.bevformer.ffn',
 'gac.bevformer_geometry': 'gac.models.bevformer.geometry',
 'gac.bevformer_history_pipeline': 'gac.models.bevformer.history_pipeline',
 'gac.bevformer_motion': 'gac.models.bevformer.motion',
 'gac.bevformer_partition': 'gac.workflows.bevformer_partition',
 'gac.bevformer_postprocess': 'gac.models.bevformer.postprocess',
 'gac.bevformer_query_init': 'gac.models.bevformer.query_init',
 'gac.bevformer_refinement': 'gac.models.bevformer.refinement',
 'gac.bevformer_runtime': 'gac.models.bevformer.runtime',
 'gac.bevformer_spatial': 'gac.models.bevformer.spatial',
 'gac.bevformer_step_pipeline': 'gac.models.bevformer.step_pipeline',
 'gac.bevformer_temporal': 'gac.models.bevformer.temporal',
 'gac.capability': 'gac.compiler.capability',
 'gac.cli': 'gac.tools.cli',
 'gac.compose': 'gac.compiler.compose',
 'gac.dispatch_runtime': 'gac.execution.dispatch',
 'gac.errors': 'gac.core.errors',
 'gac.frontends.bevformer_backbone': 'gac.models.bevformer.frontends.backbone',
 'gac.frontends.bevformer_export': 'gac.models.bevformer.frontends.export',
 'gac.frontends.bevformer_graph': 'gac.models.bevformer.frontends.graph',
 'gac.frontends.bevformer_weights': 'gac.models.bevformer.frontends.weights',
 'gac.golden': 'gac.validation.golden',
 'gac.ir': 'gac.core.ir',
 'gac.memory': 'gac.compiler.memory',
 'gac.mlir_async_tiled_matmul': 'gac.targets.cuda.async_tiled_matmul',
 'gac.mlir_cublas_ptx_artifact': 'gac.artifacts.cuda_cublas_ptx',
 'gac.mlir_cublas_ptx_benchmark': 'gac.tuning.cuda_cublas_ptx',
 'gac.mlir_cublas_ptx_runtime': 'gac.execution.mlir_cublas_ptx',
 'gac.mlir_ptx_artifact': 'gac.artifacts.cuda_ptx',
 'gac.mlir_ptx_autotune': 'gac.tuning.cuda_ptx',
 'gac.mlir_ptx_runtime': 'gac.execution.mlir_ptx',
 'gac.mlir_runtime': 'gac.execution.mlir_cuda',
 'gac.mlir_tiled_ffn': 'gac.targets.cuda.tiled_ffn',
 'gac.mlir_tiled_ffn_artifact': 'gac.artifacts.cuda_tiled_ffn',
 'gac.mlir_tiled_ffn_runtime': 'gac.execution.mlir_tiled_ffn',
 'gac.mlir_tiled_matmul': 'gac.targets.cuda.tiled_matmul',
 'gac.mlir_toolchain': 'gac.toolchains.mlir',
 'gac.ops': 'gac.core.ops',
 'gac.package': 'gac.artifacts.hxp',
 'gac.partition': 'gac.compiler.partition',
 'gac.passes': 'gac.compiler.passes',
 'gac.profiler': 'gac.execution.profiler',
 'gac.runtime': 'gac.execution.cuda',
 'gac.transformer': 'gac.models.common.transformer',
 'gac.uniad': 'gac.models.uniad',
 'gac.uniad.artifact': 'gac.models.uniad.artifact',
 'gac.uniad.checkpoint': 'gac.models.uniad.checkpoint',
 'gac.uniad.contract': 'gac.models.uniad.contract',
 'gac.version': 'gac.core.version',
 'gac.weights': 'gac.artifacts.weights'}

class _AliasLoader(Loader):
    def __init__(self, alias: str, target: str):
        self.alias = alias
        self.target = target

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        sys.modules[self.alias] = import_module(self.target)

    def get_code(self, fullname):
        # Preserve `python -m gac.cli` as well as normal import compatibility.
        source = "import runpy\nrunpy.run_module(" + repr(self.target) + ", run_name='__main__')\n"
        return compile(source, "<legacy module " + fullname + ">", "exec")

    def get_filename(self, fullname):
        # runpy uses spec.origin as argv[0]; argparse needs a real filename.
        spec = find_spec(self.target)
        if spec is None or spec.origin is None:
            raise ImportError(f"cannot locate canonical module {self.target!r}")
        return spec.origin


class _LegacyModuleFinder(MetaPathFinder):
    _gac_legacy_alias_finder = True

    def find_spec(self, fullname, path=None, target=None):
        canonical = LEGACY_MODULE_ALIASES.get(fullname)
        if canonical is None:
            return None
        return spec_from_loader(fullname, _AliasLoader(fullname, canonical),
                                is_package=fullname == "gac.uniad")


def install_legacy_imports() -> None:
    if not any(getattr(finder, "_gac_legacy_alias_finder", False) for finder in sys.meta_path):
        sys.meta_path.insert(0, _LegacyModuleFinder())
