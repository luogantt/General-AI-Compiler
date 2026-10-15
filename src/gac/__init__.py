"""General AI Compiler: a compiler-runtime co-design research prototype."""

from __future__ import annotations

__version__ = "0.1.0"

# Keep the old flat API lazy while all implementation lives in layered packages.
from ._compat import install_legacy_imports

install_legacy_imports()

