"""UniAD compiler interchange contracts and checkpoint adapters.

The modules in this package intentionally remain importable on Python 3.9 with
NumPy 1.22.  This lets the data-only exporter run in the official UniAD 2.0
environment while the compiler and CUDA runtimes stay in GAC's Python 3.11+
environment.
"""

from .artifact import (
    ArtifactError,
    BUNDLE_FORMAT,
    BUNDLE_SCHEMA_VERSION,
    UniADFFNBundle,
    load_bundle,
    write_bundle,
)
from .checkpoint import (
    CANONICAL_WEIGHT_SHAPES,
    CheckpointContractError,
    encoder_ffn_checkpoint_keys,
    extract_encoder_ffn_weights,
)
from .contract import (
    CANONICAL_WEIGHT_NAMES,
    CONTRACT_KIND,
    CONTRACT_SCHEMA_VERSION,
    ContractError,
    SAMPLE_NAMES,
    UniADFFNContract,
)

__all__ = [
    "ArtifactError",
    "BUNDLE_FORMAT",
    "BUNDLE_SCHEMA_VERSION",
    "CANONICAL_WEIGHT_NAMES",
    "CANONICAL_WEIGHT_SHAPES",
    "CONTRACT_KIND",
    "CONTRACT_SCHEMA_VERSION",
    "CheckpointContractError",
    "ContractError",
    "SAMPLE_NAMES",
    "UniADFFNBundle",
    "UniADFFNContract",
    "encoder_ffn_checkpoint_keys",
    "extract_encoder_ffn_weights",
    "load_bundle",
    "write_bundle",
]
