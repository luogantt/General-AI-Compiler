#!/usr/bin/env python3
"""Verify the independent model bundle without loading ONNX or importing CUDA."""

import argparse
import hashlib
import json
from pathlib import Path


def verify(manifest_path):
    manifest_path = Path(manifest_path).resolve(strict=True)
    asset_root = manifest_path.parent
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = manifest.get("files")
    if not isinstance(entries, list) or not entries:
        raise ValueError("manifest.files must be a nonempty list")
    seen = set()
    results = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("Each files entry must be an object")
        relative = entry.get("path")
        expected_size = entry.get("bytes")
        expected_hash = entry.get("sha256")
        if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
            raise ValueError("Each asset path must be a nonempty relative path")
        path = (asset_root / relative).resolve(strict=True)
        if not path.is_relative_to(asset_root) or path == manifest_path:
            raise ValueError(f"Invalid asset path: {relative}")
        if path in seen:
            raise ValueError(f"Duplicate asset path: {relative}")
        seen.add(path)
        if type(expected_size) is not int or expected_size < 0:
            raise ValueError(f"Invalid byte count: {relative}")
        if not isinstance(expected_hash, str) or len(expected_hash) != 64:
            raise ValueError(f"Invalid SHA-256: {relative}")
        try:
            int(expected_hash, 16)
        except ValueError as error:
            raise ValueError(f"Invalid SHA-256: {relative}") from error
        if not path.is_file():
            raise ValueError(f"Asset is not a file: {relative}")
        actual_size = path.stat().st_size
        if actual_size != expected_size:
            raise ValueError(f"Size mismatch for {relative}: {actual_size} != {expected_size}")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 << 20), b""):
                digest.update(block)
        actual_hash = digest.hexdigest()
        if actual_hash != expected_hash.lower():
            raise ValueError(f"SHA-256 mismatch: {relative}")
        results.append({"path": relative, "bytes": actual_size, "sha256": actual_hash})
    return {"passed": True, "files": results, "scope": "File integrity only, not model accuracy"}


def main():
    project_dir = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=project_dir / "models" / "bevformer-base" / "manifest.json",
    )
    args = parser.parse_args()
    print(json.dumps(verify(args.manifest), indent=2))


if __name__ == "__main__":
    main()
