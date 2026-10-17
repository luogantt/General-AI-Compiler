#!/usr/bin/env python3
"""Fetch pinned upstream sources; never install packages or execute their code."""

import argparse
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
LOCK = Path(__file__).with_name("sources.json")


def git(repo, *args):
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True, text=True, stdout=subprocess.PIPE,
    ).stdout.strip()


def prepare(source, cache, fetch):
    repo = cache / source["name"]
    if not repo.exists():
        if not fetch:
            raise RuntimeError(f"Missing {repo}; run with --fetch first")
        cache.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "--depth", "1", "--no-tags", "--no-checkout",
             "--filter=blob:none", "--sparse", source["url"], str(repo)],
            check=True,
        )
        git(repo, "fetch", "--depth", "1", "origin", source["commit"])
        if source.get("sparse_paths"):
            git(repo, "sparse-checkout", "set", *source["sparse_paths"])
        else:
            git(repo, "sparse-checkout", "disable")
        git(repo, "checkout", "--detach", source["commit"])

    # Existing checkouts are inspected, never reset or silently patched.
    if Path(git(repo, "rev-parse", "--show-toplevel")).resolve() != repo.resolve():
        raise RuntimeError(f"Not an independent Git checkout: {repo}")
    if git(repo, "rev-parse", "HEAD") != source["commit"]:
        raise RuntimeError(f"Unexpected revision in {repo}; choose a fresh --cache")
    if git(repo, "status", "--porcelain"):
        raise RuntimeError(f"Modified upstream checkout: {repo}; preserve it and use a fresh --cache")
    for relative in source["required_files"]:
        if not (repo / relative).is_file():
            raise RuntimeError(f"Missing upstream entry: {repo / relative}")
    print(f"Verified {source['name']}: {source['commit']}")
    return repo


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--fetch", action="store_true", help="Download missing checkouts")
    action.add_argument("--check", action="store_true", help="Check existing sources (default)")
    parser.add_argument("--cache", type=Path, default=ROOT / "build" / "external")
    args = parser.parse_args()
    sources = json.loads(LOCK.read_text(encoding="utf-8"))["sources"]
    repos = {s["name"]: prepare(s, args.cache.resolve(), args.fetch) for s in sources}
    patch = repos["DL4AGX"] / "AV-Solutions/bevformer-int8-eq/bevformer_trt10.patch"
    git(repos["BEVFormer_tensorrt"], "apply", "--check", str(patch))
    print("TensorRT 10 patch applies cleanly (check only; not applied).")
    print("Source preparation only. No engine build, inference, or speed validation performed.")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(1)
