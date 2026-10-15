#!/usr/bin/env python3
"""Source-checkout entry point for the single-device Vortex graph compiler."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gac.tools.build_vortex_graph import main

if __name__ == "__main__":
    raise SystemExit(main())
