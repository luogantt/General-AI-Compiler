#!/usr/bin/env python3
"""Archive small text evidence; exclude models, binaries and weight files."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


def digest(path):
    checksum = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--llvm-bin", type=Path, required=True)
    parser.add_argument("--compiler-bin", type=Path, required=True)
    parser.add_argument("--host-results", type=Path,
                        help="Override host report when the final suite uses another fixture directory")
    parser.add_argument("--extra-fixtures", type=Path)
    parser.add_argument("--cse-fixtures", type=Path)
    args = parser.parse_args()
    repository = Path(__file__).resolve().parents[2]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    def copy(source, target):
        destination = output / target
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)

    copy(args.host_results or args.fixtures / "results.json", "native-results.json")
    copy(args.fixtures / "vortex-object-results.json", "vortex-results.json")
    if args.extra_fixtures:
        copy(args.extra_fixtures / "selected-results.json", "native-extra-results.json")
    if args.cse_fixtures:
        copy(args.cse_fixtures / "selected-results.json", "native-cse-results.json")
    for case in ("matmul_bias_relu", "conv_bias_relu", "softmax"):
        artifact = args.fixtures / case / "vortex-simx"
        copy(artifact / "outputs/runtime-report.json", f"simx/{case}.json")
        copy(artifact / "run.log", f"simx/{case}.log")
        copy(args.fixtures / case / "vortex/symbols.txt", f"symbols/{case}.txt")
    matmul = args.fixtures / "matmul_bias_relu/vortex-simx"
    for filename in ("01-import.mlir", "02-optimized.mlir", "03-device.mlir",
                     "kernel.mlir", "kernel.ll", "plan.json"):
        copy(matmul / filename, f"matmul-example/{filename}")

    evidence = {"branch": subprocess.check_output(
        ["git", "-C", str(repository), "branch", "--show-current"], text=True).strip(),
        "base_commit": subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True).strip(),
        "note": "Validation includes uncommitted native implementation; source hashes identify it.",
        "compiler_sources_sha256": {}, "tools": {}, "simx_artifacts_sha256": {}}
    for source in sorted((repository / "native").rglob("*")):
        if source.is_file() and source.suffix in (".cpp", ".h", ".py", ".sh", ".mk", ".txt"):
            evidence["compiler_sources_sha256"][str(source.relative_to(repository))] = digest(source)
    for name in ("mlir-opt", "mlir-translate", "clang", "llc"):
        binary = args.llvm_bin / name
        evidence["tools"][name] = {"path": str(binary), "sha256": digest(binary),
            "version": subprocess.check_output([str(binary), "--version"], text=True).strip()}
    for name in ("gac-compile", "gac-opt", "gac-vortex-run"):
        binary = args.compiler_bin / name
        evidence["tools"][name] = {"path": str(binary), "sha256": digest(binary)}
    for case in ("matmul_bias_relu", "conv_bias_relu", "softmax"):
        artifact = args.fixtures / case / "vortex-simx"
        evidence["simx_artifacts_sha256"][case] = {
            filename: digest(artifact / filename) for filename in ("kernel.elf", "kernel.vxbin")}
    (output / "provenance.json").write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    print(f"Archived text evidence in {output}")


if __name__ == "__main__":
    main()
