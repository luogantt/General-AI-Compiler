"""Checkout wrapper for :mod:`gac.tools.build_bevformer_vortex`."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gac.tools.build_bevformer_vortex import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
