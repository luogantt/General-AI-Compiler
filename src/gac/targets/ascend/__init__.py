"""Ascend target discovery utilities.

The physical compiler/runtime is intentionally not exported here yet.  Phase 1
only provides a fail-closed environment probe while the native compiler learns
to emit reviewable TTIR artifacts for an Ascend hand-off.
"""

from .probe import probe_ascend_environment

__all__ = ["probe_ascend_environment"]
