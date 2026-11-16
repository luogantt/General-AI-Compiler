#!/usr/bin/env python3
"""Check or apply the pinned GalaxeaVLA TorchNPU source overlay.

The default mode is deliberately read-only: it validates the upstream commit,
the bundled patch digest and ``git apply --check``.  ``--apply`` additionally
requires a completely clean upstream worktree, repeats the check, and only then
applies the patch without staging or committing it.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import subprocess
import sys
from typing import Sequence


EXPECTED_UPSTREAM_REVISION = "89f2322b4ad016e192437adc1a2c253b05bab246"
EXPECTED_PATCH_SHA256 = "3cfeedaa69f7d965c5601369082e3314c5f95b11b8ecc849675b8c132b8e047d"
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PATCH_PATH = (
    REPOSITORY_ROOT
    / "integrations"
    / "galaxeavla"
    / "patches"
    / "0001-ascend-torchnpu-runtime.patch"
)


class OverlayError(RuntimeError):
    """Raised when applying the overlay would be unsafe or non-reproducible."""


def _run_git(repo: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        ["git", "-C", str(repo), *arguments],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode:
        detail = result.stderr.strip() or result.stdout.strip() or "git command failed"
        raise OverlayError(f"git {' '.join(arguments)}: {detail}")
    return result


def resolve_worktree(path: Path) -> Path:
    """Return the canonical Git worktree root for ``path``."""

    candidate = path.expanduser().resolve()
    if not candidate.is_dir():
        raise OverlayError(f"upstream path is not a directory: {candidate}")
    result = _run_git(candidate, "rev-parse", "--show-toplevel")
    return Path(result.stdout.strip()).resolve()


def verify_upstream_revision(worktree: Path) -> str:
    revision = _run_git(worktree, "rev-parse", "HEAD").stdout.strip().lower()
    if revision != EXPECTED_UPSTREAM_REVISION:
        raise OverlayError(
            "unexpected GalaxeaVLA revision: "
            f"expected {EXPECTED_UPSTREAM_REVISION}, observed {revision}"
        )
    return revision


def verify_patch_asset(patch_path: Path = PATCH_PATH) -> str:
    """Validate that the integration asset is the reviewed immutable patch."""

    if not patch_path.is_file():
        raise OverlayError(f"overlay patch is missing: {patch_path}")
    digest = hashlib.sha256(patch_path.read_bytes()).hexdigest()
    if digest != EXPECTED_PATCH_SHA256:
        raise OverlayError(
            f"overlay patch SHA-256 mismatch: expected {EXPECTED_PATCH_SHA256}, observed {digest}"
        )
    return digest


def worktree_changes(worktree: Path) -> list[str]:
    output = _run_git(
        worktree,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    ).stdout
    return [line for line in output.splitlines() if line]


def check_overlay(worktree: Path, patch_path: Path = PATCH_PATH) -> None:
    """Ask Git to validate every patch hunk without changing the worktree."""

    _run_git(
        worktree,
        "apply",
        "--check",
        "--whitespace=error-all",
        str(patch_path.resolve()),
    )


def apply_overlay(worktree: Path, patch_path: Path = PATCH_PATH) -> None:
    """Apply the patch only after clean-tree and dry-run validation."""

    changes = worktree_changes(worktree)
    if changes:
        preview = "\n".join(changes[:20])
        suffix = "\n..." if len(changes) > 20 else ""
        raise OverlayError(
            "refusing to apply over a dirty GalaxeaVLA worktree; preserve or commit "
            f"the existing changes first:\n{preview}{suffix}"
        )
    check_overlay(worktree, patch_path)
    _run_git(worktree, "apply", "--whitespace=error-all", str(patch_path.resolve()))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "upstream",
        type=Path,
        help=f"GalaxeaVLA worktree pinned at {EXPECTED_UPSTREAM_REVISION}",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="apply the overlay after a mandatory dry run; default only checks it",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        worktree = resolve_worktree(args.upstream)
        revision = verify_upstream_revision(worktree)
        digest = verify_patch_asset()
        if args.apply:
            apply_overlay(worktree)
            print(f"Applied TorchNPU overlay to {worktree}")
        else:
            check_overlay(worktree)
            print(f"Overlay check passed for {worktree}; no files were changed")
        print(f"Upstream revision: {revision}")
        print(f"Patch SHA-256: {digest}")
        return 0
    except OverlayError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
