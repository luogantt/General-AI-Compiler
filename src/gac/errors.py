"""Error types shared by the compiler and runtime prototype."""

from __future__ import annotations


class GACError(Exception):
    """Base class for expected, user-facing GAC failures."""


class ValidationError(GACError):
    """Raised when an IR object violates its structural invariants."""


class ShapeInferenceError(GACError):
    """Raised when an operation's result shape or dtype cannot be inferred."""


class EvaluationError(GACError):
    """Raised when the reference evaluator cannot execute an operation."""


class PassError(GACError):
    """Raised when an optimization pass fails."""


class BackendError(GACError):
    """A target cannot compile or execute the requested program."""


class DeviceUnavailableError(BackendError):
    """The requested physical device or its dependencies are unavailable."""
