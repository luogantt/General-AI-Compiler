"""Build the registration-only TorchNPU eager bridge; no NPU computation.

Run in the sourced CANN/PyTorch environment, after installing the separately
built custom OPP package. This script does not build/install/overwrite that
package. Load the resulting shared object with torch.ops.load_library(path),
not Python import. Kernel execution/precision/stream testing is a separate step.

Example:
  python build_bridge.py --opapi-include-dir /custom/op_api/include \
    --opapi-lib-dir /custom/op_api/lib --custom-opp-path /custom --load
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys


EXTENSION_NAME = "gac_g05_ascendc_qknorm_rope_bridge"
API_HEADER = "aclnn_gac_qk_norm_rope.h"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--opapi-include-dir", type=Path, required=True,
                        help=f"Directory containing the generated {API_HEADER}")
    parser.add_argument("--opapi-lib-dir", type=Path, required=True,
                        help="Directory containing the matching libcust_opapi.so")
    parser.add_argument("--custom-opp-path", type=Path,
                        help="Installed custom vendor root; prepended before importing TorchNPU")
    parser.add_argument("--cann-include-dir", type=Path,
                        help="Optional CANN include root (normally ASCEND_HOME_PATH/include)")
    parser.add_argument("--cann-lib-dir", type=Path,
                        help="CANN library root with libnnopbase.so/libascendcl.so (normally ASCEND_HOME_PATH/lib64)")
    parser.add_argument("--build-dir", type=Path, default=Path("build/ascendc_qknorm_rope_bridge"))
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--load", action="store_true",
                        help="After building, load the shared object and check dispatcher registration only")
    args = parser.parse_args(argv)
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    return args


def require_file(path: Path, description: str) -> Path:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Missing {description}: {resolved}")
    return resolved


def build(args) -> dict:
    if sys.platform != "linux":
        raise RuntimeError("Build on the Linux Ascend host with its installed TorchNPU/CANN toolchain")
    source = require_file(Path(__file__).with_name("torch_bridge.cpp"), "bridge source")
    include_dir = args.opapi_include_dir.resolve()
    lib_dir = args.opapi_lib_dir.resolve()
    api_header = require_file(include_dir / API_HEADER, "generated ACLNN header")
    opapi_library = require_file(lib_dir / "libcust_opapi.so", "custom ACLNN library")

    # CANN/TorchNPU discover custom paths during library startup.
    # Set this before the first torch_npu import; leave the existing path entries
    # intact and never rewrite the user's installed environment setup scripts.
    if args.custom_opp_path is not None:
        custom_root = args.custom_opp_path.resolve()
        if not custom_root.is_dir():
            raise NotADirectoryError(custom_root)
        previous = os.environ.get("ASCEND_CUSTOM_OPP_PATH", "")
        entries = [str(custom_root), *(part for part in previous.split(os.pathsep) if part)]
        os.environ["ASCEND_CUSTOM_OPP_PATH"] = os.pathsep.join(dict.fromkeys(entries))
    if "torch_npu" in sys.modules and args.custom_opp_path is not None:
        raise RuntimeError("Start a fresh builder process: custom OPP path must be set before TorchNPU import")

    import torch
    import torch_npu
    from setuptools import Distribution
    from torch.utils.cpp_extension import BuildExtension
    from torch_npu.utils.cpp_extension import NpuExtension

    npu_root = Path(torch_npu.__file__).resolve().parent
    torch_headers = [require_file(npu_root / "include" / name, "TorchNPU exported bridge interface")
                     for name in ("torch_npu/csrc/framework/OpCommand.h",
                                  "torch_npu/csrc/core/npu/NPUStream.h",
                                  "torch_npu/csrc/aten/CustomFunctions.h")]
    include_dirs = [str(include_dir)]
    cann_include = args.cann_include_dir
    if cann_include is None and os.environ.get("ASCEND_HOME_PATH"):
        cann_include = Path(os.environ["ASCEND_HOME_PATH"]) / "include"
    if cann_include is not None:
        cann_include = cann_include.resolve()
        if not cann_include.is_dir():
            raise NotADirectoryError(cann_include)
        include_dirs.append(str(cann_include))
    cann_lib_dir = args.cann_lib_dir
    if cann_lib_dir is None and os.environ.get("ASCEND_HOME_PATH"):
        cann_lib_dir = Path(os.environ["ASCEND_HOME_PATH"]) / "lib64"
    if cann_lib_dir is None:
        raise ValueError("Pass --cann-lib-dir or source CANN set_env.sh to set ASCEND_HOME_PATH")
    cann_lib_dir = cann_lib_dir.resolve()
    cann_libraries = [require_file(cann_lib_dir / name, "CANN direct ACL tensor/runtime library")
                      for name in ("libnnopbase.so", "libascendcl.so")]

    build_dir = args.build_dir.resolve()
    build_dir.mkdir(parents=True, exist_ok=True)
    # Pin the custom API and CANN libraries. Keep -z,defs: missing private/runtime
    # symbols must fail the build, never be hidden until the first NPU request.
    extension = NpuExtension(
        name=EXTENSION_NAME,
        sources=[str(source)],
        include_dirs=include_dirs,
        library_dirs=[str(lib_dir), str(cann_lib_dir)],
        extra_compile_args=["-O3", "-std=c++17", "-fPIC"],
        extra_link_args=["-Wl,--no-as-needed", str(opapi_library),
                         *(str(path) for path in cann_libraries), "-Wl,--as-needed",
                         f"-Wl,-rpath,{lib_dir}", f"-Wl,-rpath,{cann_lib_dir}", "-Wl,-z,defs"],
    )
    distribution = Distribution({
        "name": EXTENSION_NAME,
        "ext_modules": [extension],
        "cmdclass": {"build_ext": BuildExtension.with_options(use_ninja=False, no_python_abi_suffix=True)},
    })
    command = distribution.get_command_obj("build_ext")
    command.build_lib = str(build_dir / "lib")
    command.build_temp = str(build_dir / "objects")
    command.parallel = args.jobs
    command.ensure_finalized()
    command.run()
    artifact = require_file(Path(command.get_ext_fullpath(EXTENSION_NAME)), "built bridge shared object")

    registered = False
    if args.load:
        torch.ops.load_library(str(artifact))
        registered = bool(torch._C._dispatch_has_kernel_for_dispatch_key(
            "gac_g05::qk_norm_rope", "PrivateUse1"))
        if not registered:
            raise RuntimeError("Bridge loaded but the NPU dispatcher kernel was not registered")
    result = {
        "status": "built", "artifact": str(artifact), "artifact_sha256": sha256_file(artifact),
        "schema": "gac_g05::qk_norm_rope(Tensor x, Tensor weight, Tensor cos, Tensor sin, int head_stride, float epsilon) -> Tensor",
        "device_kernel_registration_checked": args.load, "device_kernel_registered": registered,
        "npu_execution_performed": False, "graph_compatibility_validated": False,
        "torch": torch.__version__, "torch_npu": torch_npu.__version__,
        "python": platform.python_version(), "machine": platform.machine(),
        "cxx11_abi": bool(torch._C._GLIBCXX_USE_CXX11_ABI),
        "inputs": {"source": {"path": str(source), "sha256": sha256_file(source)},
                   "api_header": {"path": str(api_header), "sha256": sha256_file(api_header)},
                   "opapi_library": {"path": str(opapi_library), "sha256": sha256_file(opapi_library)},
                   "torch_npu_headers": [{"path": str(path), "sha256": sha256_file(path)} for path in torch_headers],
                   "cann_libraries": [{"path": str(path), "sha256": sha256_file(path)} for path in cann_libraries]},
        "runtime_requirements": {
            "custom_opp_path_before_torch_npu_import": os.environ.get("ASCEND_CUSTOM_OPP_PATH", ""),
            "opapi_lib_dir": str(lib_dir), "source_cann_set_env": True,
            "load": "import torch, torch_npu; torch.ops.load_library(artifact)",
            "synchronization": "Direct ACLNN preparation/launch inside exported OpCommand::RunOpApi on current NPU stream; no per-call synchronize",
            "workspace_contract": "zero only; reject nonzero workspace, no callback ATen allocation",
            "physical_format": "unblocked ND/NCHW only; checked for every input and the fresh output",
        },
    }
    manifest = build_dir / "bridge-build.json"
    manifest.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    result["manifest"] = str(manifest)
    return result


def main(argv=None):
    print(json.dumps(build(parse_args(argv)), indent=2))


if __name__ == "__main__":
    main()
