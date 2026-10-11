#!/usr/bin/env python3
"""Checkout wrapper for :mod:`gac.tools.uniad_make_synthetic_bundle`."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gac.tools.uniad_make_synthetic_bundle import main  # noqa: E402


if __name__ == "__main__":
    main()
