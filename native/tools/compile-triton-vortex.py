#!/usr/bin/env python3
"""Build one Vortex module from native GAC and Triton dispatches.

Graph semantics, fusion and eligibility are decided by C++ MLIR in gac-compile.
This driver only invokes compilers, links their LLVM IR and records artifacts.
Run with the Python environment containing the installed triton-vortex plugin.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import time


def require_file(path: Path, description: str) -> Path:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"Missing {description}: {path}")
    return path


def power_of_two(text: str) -> int:
    value = int(text)
    if value <= 0 or value & (value - 1):
        raise argparse.ArgumentTypeError("expected a positive power of two")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_compiler(value: str | None) -> Path:
    if not value:
        raise ValueError("Pass --compiler or set GAC_NATIVE_BUILD to the native build directory")
    path = Path(value).expanduser()
    if path.is_dir():
        for relative in ("gac-compile", "bin/gac-compile"):
            if (path / relative).is_file():
                return (path / relative).resolve()
    return require_file(path, "gac-compile")


def find_makefile(root: str | None) -> Path:
    if root:
        return require_file(Path(root) / "tools/Vortex.mk", "Triton/Vortex Makefile")
    spec = importlib.util.find_spec("triton_vortex")
    if spec is None or spec.origin is None:
        raise ValueError("Install the Triton/Vortex plugin in this Python environment first")
    source_root = Path(spec.origin).resolve().parents[2]
    for path in (source_root / "tools/Vortex.mk",
                 Path(sys.prefix) / "share/triton-vortex/tools/Vortex.mk"):
        if path.is_file():
            return path
    raise ValueError("Cannot find tools/Vortex.mk; pass --triton-backend-root")


def run_step(name: str, command: list[str | Path], output: Path,
             steps: list[dict], *, result_file: Path | None = None) -> None:
    command = [str(item) for item in command]
    print(f"[{name}] {command[0]}", flush=True)
    started = time.perf_counter()
    result = subprocess.run(command, cwd=output, capture_output=True, text=True)
    log = output / f"{name}.log"
    log.write_text(result.stdout + result.stderr, encoding="utf-8")
    steps.append({"name": name, "seconds": time.perf_counter() - started,
                  "command": command, "returncode": result.returncode,
                  "log": log.name})
    if result.returncode:
        raise RuntimeError(f"{name} failed ({result.returncode}); see {log}\n{result.stderr[-4000:]}")
    if result_file is not None:
        result_file.write_text(result.stdout, encoding="utf-8")


def triton_dispatches(plan: dict, source_dir: Path, args: argparse.Namespace):
    """Validate the compiler/runtime contract without rewriting a dispatch."""
    names = set()
    tensors = {tensor.get("name"): tensor for tensor in plan.get("tensors", [])}
    if None in tensors or len(tensors) != len(plan.get("tensors", [])):
        raise ValueError("Tensor manifest contains a missing or duplicate name")
    for dispatch in plan["dispatches"]:
        if "triton" not in dispatch:
            if dispatch.get("lowering") == "linalg-to-triton":
                raise ValueError(
                    f"{dispatch.get('kernel', '<unnamed>')}: Triton dispatch is missing its launch contract")
            continue
        config = dispatch["triton"]
        name = dispatch["kernel"]
        filename = config["ttir_file"]
        if name in names or filename != f"{name}.ttir" or Path(filename).name != filename:
            raise ValueError(f"Invalid or duplicate Triton artifact name: {filename!r}")
        names.add(name)
        if dispatch.get("lowering") != "linalg-to-triton":
            raise ValueError(f"Unexpected lowering for Triton dispatch {name}")
        expected = {"num_warps": args.num_warps, "lanes": args.lanes,
                    "block_size": args.num_warps * args.lanes}
        kind = config.get("kind")
        if kind == "pointwise":
            expected.update(
                block_elements=args.block,
                grid_size=(dispatch["elements"] + args.block - 1) // args.block)
        elif kind == "softmax":
            rows, width, n, tile = (config.get("rows"), config.get("width"),
                                    config.get("n"), config.get("tile"))
            if (type(rows) is not int or not 1 <= rows <= 0x7fffffff or
                    type(width) is not int or not 1 <= width <= 1024 or
                    rows > 0x7fffffff // width or
                    type(n) is not int or n != width or
                    type(tile) is not int or not 16 <= tile <= 1024 or
                    tile & (tile - 1) or
                    tile != max(16, 1 << (width - 1).bit_length())):
                raise ValueError(
                    f"{name}: unsupported Softmax rows/width/N/tile: "
                    f"rows={rows!r}, width={width!r}, n={n!r}, "
                    f"tile={tile!r}")
            if (dispatch.get("elements") != rows * width or
                    len(dispatch.get("inputs", [])) != 1 or
                    dispatch["inputs"][0] == dispatch.get("output")):
                raise ValueError(f"{name}: Softmax manifest shape/ABI contract disagrees")
            input_tensor = tensors.get(dispatch["inputs"][0])
            output_tensor = tensors.get(dispatch.get("output"))
            if (not input_tensor or input_tensor.get("dtype") != "float32" or
                    not isinstance(input_tensor.get("shape"), list) or
                    len(input_tensor["shape"]) < 1 or
                    input_tensor["shape"][-1] != width or
                    any(type(dim) is not int or dim <= 0
                        for dim in input_tensor["shape"]) or
                    input_tensor.get("bytes") != rows * width * 4):
                raise ValueError(
                    f"{name}: Softmax input must be static FP32 [...,width]")
            prefix_rows = 1
            for dim in input_tensor["shape"][:-1]:
                prefix_rows *= dim
            if prefix_rows != rows:
                raise ValueError(
                    f"{name}: Softmax input prefix product must equal rows")
            if (not output_tensor or output_tensor.get("dtype") != "float32" or
                    output_tensor.get("shape") != input_tensor["shape"] or
                    output_tensor.get("bytes") != rows * width * 4):
                raise ValueError(
                    f"{name}: Softmax output must exactly match input")
            expected.update(grid_size=rows,
                            abi="vortex-u64-pointers-i32-n")
        elif kind == "layer_norm":
            rows, width, n, tile = (config.get("rows"), config.get("width"),
                                    config.get("n"), config.get("tile"))
            epsilon_bits = config.get("epsilon_f32_bits")
            epsilon = (struct.unpack("<f", struct.pack("<I", epsilon_bits))[0]
                       if type(epsilon_bits) is int and
                       0 <= epsilon_bits <= 0xffffffff else float("nan"))
            if (type(rows) is not int or not 1 <= rows <= 0x7fffffff or
                    type(width) is not int or not 1 <= width <= 1024 or
                    rows > 0x7fffffff // width or
                    type(n) is not int or n != width or
                    type(tile) is not int or not 16 <= tile <= 1024 or
                    tile & (tile - 1) or
                    tile != max(16, 1 << (width - 1).bit_length()) or
                    type(epsilon_bits) is not int or
                    not 0 <= epsilon_bits <= 0xffffffff or
                    not math.isfinite(epsilon) or epsilon <= 0):
                raise ValueError(
                    f"{name}: unsupported LayerNorm rows/width/N/tile/epsilon: "
                    f"rows={rows!r}, width={width!r}, n={n!r}, "
                    f"tile={tile!r}, epsilon_f32_bits={epsilon_bits!r}")
            input_names = dispatch.get("inputs", [])
            output_name = dispatch.get("output")
            if (len(input_names) != 3 or len(set(input_names)) != 3 or
                    output_name in input_names or
                    dispatch.get("elements") != rows * width):
                raise ValueError(
                    f"{name}: LayerNorm manifest pointer ABI contract disagrees")
            x = tensors.get(input_names[0])
            scale = tensors.get(input_names[1])
            bias = tensors.get(input_names[2])
            output = tensors.get(output_name)
            if (not x or x.get("dtype") != "float32" or
                    not isinstance(x.get("shape"), list) or
                    len(x["shape"]) < 2 or x["shape"][-1] != width or
                    any(type(dim) is not int or dim <= 0 for dim in x["shape"]) or
                    x.get("bytes") != rows * width * 4):
                raise ValueError(
                    f"{name}: LayerNorm X must be static FP32 [...,width]")
            prefix_rows = 1
            for dim in x["shape"][:-1]:
                prefix_rows *= dim
            if prefix_rows != rows:
                raise ValueError(
                    f"{name}: LayerNorm X prefix product must equal rows")
            if (not output or output.get("dtype") != "float32" or
                    output.get("shape") != x["shape"] or
                    output.get("bytes") != rows * width * 4):
                raise ValueError(
                    f"{name}: LayerNorm Y must exactly match X")
            for label, tensor in (("Scale", scale), ("Bias", bias)):
                if (not tensor or tensor.get("dtype") != "float32" or
                        tensor.get("shape") != [width] or
                        tensor.get("bytes") != width * 4):
                    raise ValueError(
                        f"{name}: LayerNorm {label} must be FP32 vector[width]")
            expected.update(
                grid_size=rows,
                abi="vortex-u64-pointers-i32-n-f32-epsilon")
        elif kind == "reduction":
            width, tile = config.get("width"), config.get("tile")
            if (type(width) is not int or not 1 <= width <= 1024 or
                    type(tile) is not int or not 16 <= tile <= 1024 or
                    tile & (tile - 1) or tile != max(16, 1 << (width - 1).bit_length())):
                raise ValueError(
                    f"{name}: unsupported reduction width/tile: width={width!r}, tile={tile!r}")
            if dispatch.get("elements") != 1 or len(dispatch.get("inputs", [])) != 1:
                raise ValueError(f"{name}: reduction manifest scalar/ABI contract disagrees")
            input_tensor = tensors.get(dispatch["inputs"][0])
            output_tensor = tensors.get(dispatch.get("output"))
            if (not input_tensor or input_tensor.get("dtype") != "float32" or
                    input_tensor.get("shape") != [width] or
                    input_tensor.get("bytes") != width * 4):
                raise ValueError(f"{name}: reduction input must be FP32 vector[width]")
            if (not output_tensor or output_tensor.get("dtype") != "float32" or
                    output_tensor.get("shape") not in ([], [1]) or
                    output_tensor.get("bytes") != 4):
                raise ValueError(f"{name}: reduction output must be one FP32 scalar")
            expected["grid_size"] = 1
        elif kind == "matmul":
            dimensions = {key: config.get(key) for key in
                          ("m", "n", "k", "block_m", "block_n", "block_k")}
            if any(type(value) is not int or not 0 < value <= 0x7fffffff
                   for value in dimensions.values()):
                raise ValueError(f"{name}: invalid MatMul dimensions: {dimensions!r}")
            m, n, k = dimensions["m"], dimensions["n"], dimensions["k"]
            bm, bn, bk = (dimensions["block_m"], dimensions["block_n"],
                          dimensions["block_k"])
            padded_k = 1 << (k - 1).bit_length()
            if (bm != 16 or bn != 16 or not 16 <= k <= 65536 or
                    k % 16 or bk != padded_k):
                raise ValueError(f"{name}: unsupported MatMul tile contract: {dimensions!r}")
            input_names = dispatch.get("inputs", [])
            output_name = dispatch.get("output")
            if (len(input_names) != 2 or len(set(input_names)) != 2 or
                    output_name in input_names):
                raise ValueError(f"{name}: MatMul pointer ABI contract disagrees")
            a, b = (tensors.get(input_names[0]), tensors.get(input_names[1]))
            c = tensors.get(output_name)
            if not a or not b or not c:
                raise ValueError(f"{name}: MatMul tensor is missing from manifest")

            batch_keys = ("batches", "batch_mode", "a_shape", "b_shape", "c_shape")
            present = [key in config for key in batch_keys]
            if any(present) and not all(present):
                raise ValueError(f"{name}: incomplete batched MatMul contract")

            def checked_shape(value, label):
                if (not isinstance(value, list) or len(value) < 2 or
                        any(type(dim) is not int or dim <= 0 for dim in value)):
                    raise ValueError(f"{name}: invalid MatMul {label} shape {value!r}")
                elements = 1
                for dim in value:
                    if elements > 0x7fffffff // dim:
                        raise ValueError(
                            f"{name}: MatMul {label} element count exceeds signed i32")
                    elements *= dim
                return value, elements

            # Old rank-2 artifacts predate the explicit batch metadata. They are
            # accepted only when all five fields are absent and physical tensors
            # prove the original A[M,K], B[K,N], C[M,N] contract.
            if not any(present):
                batches, batch_mode = 1, "rank2"
                a_shape, b_shape, c_shape = (a.get("shape"), b.get("shape"),
                                             c.get("shape"))
            else:
                batches, batch_mode = config.get("batches"), config.get("batch_mode")
                a_shape, b_shape, c_shape = (config.get("a_shape"),
                                             config.get("b_shape"),
                                             config.get("c_shape"))
            if type(batches) is not int or not 1 <= batches <= 0x7fffffff:
                raise ValueError(f"{name}: invalid MatMul batches={batches!r}")
            a_shape, a_elements = checked_shape(a_shape, "A")
            b_shape, b_elements = checked_shape(b_shape, "B")
            c_shape, c_elements = checked_shape(c_shape, "C")

            if batch_mode == "rank2":
                valid_shapes = (batches == 1 and a_shape == [m, k] and
                                b_shape == [k, n] and c_shape == [m, n])
            elif batch_mode in ("matched-prefix", "shared-rhs"):
                prefix = c_shape[:-2]
                prefix_product = 1
                for dim in prefix:
                    if prefix_product > 0x7fffffff // dim:
                        raise ValueError(
                            f"{name}: MatMul batch prefix exceeds signed i32")
                    prefix_product *= dim
                valid_shapes = (
                    len(c_shape) >= 3 and batches == prefix_product and
                    c_shape[-2:] == [m, n] and
                    a_shape == [*prefix, m, k] and
                    (b_shape == [*prefix, k, n]
                     if batch_mode == "matched-prefix" else b_shape == [k, n]))
            else:
                valid_shapes = False
            if not valid_shapes:
                raise ValueError(
                    f"{name}: MatMul logical shapes disagree with batch_mode={batch_mode!r}")

            expected_a = batches * m * k
            expected_b = ((batches if batch_mode == "matched-prefix" else 1) *
                          k * n)
            expected_c = batches * m * n
            if max(expected_a, expected_b, expected_c) > 0x7fffffff:
                raise ValueError(f"{name}: MatMul tensor size exceeds signed i32")
            for label, tensor, logical_shape, logical_elements in (
                    ("A", a, a_shape, a_elements),
                    ("B", b, b_shape, b_elements),
                    ("C", c, c_shape, c_elements)):
                if (tensor.get("dtype") != "float32" or
                        tensor.get("shape") != logical_shape or
                        logical_elements != (expected_a if label == "A" else
                                             expected_b if label == "B" else expected_c) or
                        tensor.get("bytes") != logical_elements * 4):
                    raise ValueError(
                        f"{name}: MatMul {label} physical tensor must exactly match logical shape")
            if dispatch.get("elements") != expected_c:
                raise ValueError(f"{name}: MatMul output element count disagrees")
            grid = batches * ((m + bm - 1) // bm) * ((n + bn - 1) // bn)
            if grid > 0x7fffffff:
                raise ValueError(f"{name}: MatMul grid exceeds signed i32")
            expected["grid_size"] = grid
        else:
            raise ValueError(f"{name}: unsupported Triton dispatch kind {kind!r}")
        if kind not in ("softmax", "layer_norm"):
            expected["abi"] = "vortex-u64-pointers"
        for key, value in expected.items():
            if config.get(key) != value:
                raise ValueError(f"{name}: manifest {key}={config.get(key)!r}, expected {value!r}")
        yield dispatch, require_file(source_dir / filename, "generated TTIR")


def build(args: argparse.Namespace) -> Path:
    model = require_file(Path(args.model), "ONNX model")
    compiler = find_compiler(args.compiler)
    roots = {}
    for key in ("VOLT_ROOT", "VORTEX_BUILD"):
        if not os.environ.get(key):
            raise ValueError(f"Set {key} to the configured VOLT/Vortex build")
        roots[key] = Path(os.environ[key]).expanduser().resolve()
    volt, vortex = roots["VOLT_ROOT"], roots["VORTEX_BUILD"]
    tools = {name: require_file(volt / "bin" / name, name)
             for name in ("mlir-translate", "llvm-link", "llvm-objdump")}
    require_file(vortex / "config.mk", "configured Vortex config.mk")
    makefile = find_makefile(args.triton_backend_root)
    make = shutil.which("make")
    if make is None:
        raise ValueError("GNU make is required")
    if not 16 <= args.block <= 1024:
        raise ValueError("--block must be a power of two between 16 and 1024")
    if args.num_warps > 4 or args.lanes > 32:
        raise ValueError("This GAC path requires --num-warps <= 4 and --lanes <= 32")
    output = Path(args.output).expanduser().resolve()
    # Never reuse constants, intermediate LLVM IR or binaries from an old build.
    if output.exists():
        raise ValueError(f"Output directory already exists; choose a new --output: {output}")
    output.mkdir(parents=True)
    source_dir = output / "triton"
    source_dir.mkdir()
    started = time.perf_counter()
    report = {"status": "building", "model": str(model), "compiler": str(compiler),
              "model_sha256": sha256(model), "compiler_sha256": sha256(compiler),
              "backend_makefile": str(makefile), "backend_makefile_sha256": sha256(makefile),
              "constant_fold_profile": args.constant_fold_profile,
              "target": "vortex-rv32", "steps": [], "triton_dispatches": [],
              "timing_scope": "host compile wall time, including per-kernel packaging; not inference latency"}
    # An isolated cache makes each build inspectable and prevents a previous
    # invocation from concealing a broken compiler/toolchain installation.
    os.environ["TRITON_CACHE_DIR"] = str(output / "triton-cache")
    try:
        import triton
        from triton.backends.compiler import GPUTarget

        report["triton_version"] = triton.__version__
        run_step("gac-compile", [compiler, model, "--emit=llvm", "--target=vortex",
                 "--regular-lowering=linalg", f"--vortex-triton-dir={source_dir}",
                 f"--constant-fold-profile={args.constant_fold_profile}",
                 f"--triton-block={args.block}", f"--triton-num-warps={args.num_warps}",
                 f"--vortex-lanes={args.lanes}", "--save-stages", output,
                 "--manifest", output / "plan.json", "--constants-dir", output,
                 "-o", output / "native.mlir"], output, report["steps"])
        run_step("native-translate", [tools["mlir-translate"], "--mlir-to-llvmir",
                 output / "native.mlir", "-o", output / "native.ll"], output, report["steps"])
        plan = json.loads((output / "plan.json").read_text(encoding="utf-8"))
        llvm_files = [output / "native.ll"]
        for dispatch, ttir in triton_dispatches(plan, source_dir, args):
            name = dispatch["kernel"]
            print(f"[triton] {name}", flush=True)
            compile_started = time.perf_counter()
            kernel = triton.compile(str(ttir), target=GPUTarget("vortex", "rv32", args.lanes),
                                    options={"num_warps": args.num_warps,
                                             "enable_fp_fusion": False})
            duration = time.perf_counter() - compile_started
            if kernel.metadata.name != name:
                raise ValueError(f"Triton kernel name changed: {name} -> {kernel.metadata.name}")
            for stage in ("ttir", "vxmlir", "llir"):
                text = kernel.asm.get(stage)
                if not isinstance(text, str) or not text.strip():
                    raise ValueError(f"Triton did not produce {stage} for {name}")
                suffix = "ll" if stage == "llir" else stage
                # Preserve the C++ emitter's original TTIR alongside the IR
                # parsed by Triton; no Python code changes operator semantics.
                filename = f"{name}.triton.ttir" if stage == "ttir" else f"{name}.{suffix}"
                (source_dir / filename).write_text(text, encoding="utf-8")
            llvm_files.append(source_dir / f"{name}.ll")
            report["triton_dispatches"].append({"kernel": name, "seconds": duration,
                                                "config": dispatch["triton"]})
        report["dispatch_count"] = len(plan["dispatches"])
        report["triton_dispatch_count"] = len(report["triton_dispatches"])
        report["native_dispatch_count"] = report["dispatch_count"] - report["triton_dispatch_count"]
        run_step("llvm-link", [tools["llvm-link"], "-S", *llvm_files,
                 "-o", output / "kernel.ll"], output, report["steps"])
        run_step("vortex-package", [make, "-f", makefile, f"ROOT_DIR={vortex}",
                 f"LLVM_PATH={volt}", "kernel.vxbin"], output, report["steps"])
        require_file(output / "kernel.vxbin", "linked device executable")
        run_step("disassemble", [tools["llvm-objdump"], "-d", output / "kernel.elf"],
                 output, report["steps"], result_file=output / "kernel.dump")
        report["status"] = "success"
    except Exception as error:
        report["status"] = "failed"
        report["error"] = str(error)
        raise
    finally:
        report["total_seconds"] = time.perf_counter() - started
        (output / "compile-report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="ONNX model imported by native C++ MLIR")
    parser.add_argument("--output", required=True, help="new output directory; existing directories are rejected")
    parser.add_argument("--compiler", default=os.environ.get("GAC_NATIVE_BUILD"),
                        help="gac-compile executable or native build directory (default: GAC_NATIVE_BUILD)")
    parser.add_argument("--triton-backend-root", help="optional installed/source plugin root containing tools/Vortex.mk")
    parser.add_argument("--block", "--triton-block", type=power_of_two, default=128)
    parser.add_argument("--num-warps", "--triton-num-warps", type=power_of_two, default=1)
    parser.add_argument("--lanes", "--vortex-lanes", type=power_of_two, default=4)
    parser.add_argument("--constant-fold-profile", choices=("default", "bevformer-base"),
                        default="default", help="native MLIR constant-folding resource profile")
    args = parser.parse_args()
    try:
        output = build(args)
    except (OSError, ValueError, RuntimeError, ImportError, KeyError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(f"Vortex executable: {output / 'kernel.vxbin'}")
    print(f"Dispatch plan: {output / 'plan.json'}")
    print(f"Compile timings: {output / 'compile-report.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
