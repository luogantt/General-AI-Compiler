#!/usr/bin/env python3
"""Generate/build an isolated CANN Ascend C OPP for GacGdnDecode; never install into CANN.

Run after sourcing CANN set_env.sh. The output directory must not exist: this
keeps generated project/build files separate from sources and prior results.
The generated .run installer can subsequently be given an explicit install path.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--soc", default="Ascend910B2")
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    source = Path(__file__).resolve().parent
    destination = args.output.resolve()
    if destination.exists():
        raise FileExistsError(f"Choose a new output directory: {destination}")
    cann = Path(os.environ["ASCEND_HOME_PATH"]).resolve()
    descriptor = source / "GacGdnDecode.json"
    files = [descriptor, source / "op_host/gac_gdn_decode.cpp",
             source / "op_kernel/gac_gdn_decode.cpp",
             source / "op_kernel/gac_gdn_decode_tiling.h"]
    for path in files:
        if not path.is_file():
            raise FileNotFoundError(path)
    subprocess.run(["msopgen", "gen", "-i", str(descriptor), "-f", "pytorch",
                    "-c", "ai_core-" + args.soc, "-lan", "cpp", "-out", str(destination)], check=True)
    for path in files[1:]:
        shutil.copy2(path, destination / path.relative_to(source))
    preset_path = destination / "CMakePresets.json"
    preset = json.loads(preset_path.read_text())
    variables = preset["configurePresets"][0]["cacheVariables"]
    variables["vendor_name"]["value"] = "gacgdn"
    variables["ENABLE_TEST"]["value"] = "False"
    variables["ASCEND_CANN_PACKAGE_PATH"]["value"] = str(cann)
    preset_path.write_text(json.dumps(preset, indent=2) + "\n")
    kernel_cmake = destination / "op_kernel/CMakeLists.txt"
    kernel_cmake.write_text(kernel_cmake.read_text() +
        "\nnpu_op_kernel_options(ascendc_kernels ALL OPTIONS -ffp-contract=off)\n")
    manifest = {"soc": args.soc, "cann_path": str(cann), "vendor": "gacgdn",
                "source_sha256": {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in files}, "fp_contract": "off",
                "scope": "build only; no runtime/numerical/performance claim"}
    (destination / "source-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    subprocess.run(["cmake", "-S", str(destination), "-B", str(destination / "build_out"),
                    "--preset=default"], check=True, cwd=destination)
    subprocess.run(["cmake", "--build", str(destination / "build_out"),
                    "--target", "binary", "package", "-j", str(args.jobs)], check=True)


if __name__ == "__main__":
    main()
