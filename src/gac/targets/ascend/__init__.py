"""Ascend discovery and standard-TTIR compilation utilities.

The compiler hand-off produces ``npubin`` review artifacts with the active
Triton-Ascend stack.  It remains compile-only: GAC does not expose an Ascend
buffer planner or physical kernel-launch runtime here.
"""

from .probe import probe_ascend_environment
from .triton_compile import (
    compile_ascend_ttir,
    compile_ascend_ttir_manifest,
    validate_ascend_manifest,
)

__all__ = [
    "compile_ascend_ttir",
    "compile_ascend_ttir_manifest",
    "probe_ascend_environment",
    "validate_ascend_manifest",
]
