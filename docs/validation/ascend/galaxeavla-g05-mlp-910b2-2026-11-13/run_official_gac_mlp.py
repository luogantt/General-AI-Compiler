#!/usr/bin/env python3
"""Reproduce the official four-dispatch GAC MLP validation on Ascend.

This is a pinned validation launcher, not a general GAC runtime.  It accepts
parameterized artifact locations, but deliberately validates the exact
official plan, TTIR, constants, shapes, launch grids, and resulting npubin
hashes before executing anything.

Run after sourcing the matching CANN environment, for example:

  source /usr/local/Ascend/cann/set_env.sh
  python run_official_gac_mlp.py \
    --plan /path/to/plan.json \
    --ttir-dir /path/to/ttir \
    --constants-dir /path/to/constants \
    --output /new/output/directory \
    --cache-dir /path/to/triton-cache
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import time


OFFICIAL_PLAN_SHA256 = "08bcd8efff476a4ab3dc47d02d73c56c160840f43106fdb14c19b88454ae3e30"
OFFICIAL_TTIR_SHA256 = {
    "gac_op_0": "074834bcd9416bd6afeb04f166c1ca668857278d4b00e4c520ceddb02332677e",
    "gac_op_1": "92efc051d3e4e1d2ae399504e151a9860df83c635e2bcdbb3a38c1571849b13c",
    "gac_op_2": "6c427df053deef71f24dee11ea5bb1871e1e6c13757670e0cc564e787074be95",
    "gac_op_3": "e387097f884ef71022a892fbad9f4e0b218b2c891db2b0f6d7b3e13910cfab54",
}
OFFICIAL_CONSTANT_SHA256 = {
    "tensor_1.bin": "137c8225d248c2297f4824cac34e964e4be9701e3f03324ccfa7a3d1fd0414bb",
    "tensor_2.bin": "bae8dd59fb47a1993abf99c6fd91213256028f8855dfdb3be8f25cd071ed18eb",
    "tensor_3.bin": "a25002ba888532120116c2affe017b72752446bb43a31edd84649374ef9e8244",
}
OFFICIAL_NPUBIN = {
    "gac_op_0": (2136, "7b9ab767834aec865e787597a61488027353d598e1ffa791e75034c96734c006"),
    "gac_op_1": (2136, "5ccdcebc3cf0abc78524aea9933ce12b9774bfe584cdf5b50a8cfdbfb673ea98"),
    "gac_op_2": (3792, "9d387b3e4b1c3b3496415079fd81d68f2a74e115733c2131ee17be882c4355cc"),
    "gac_op_3": (2328, "2aa85b04add0a5aed7f5927cb8554b7eceef504a8238f2689e2b5c7c524e73ec"),
}

EXPECTED_DISPATCHES = [
    {
        "kernel": "gac_op_0", "kind": "matmul",
        "inputs": ["hidden_states", "value_1"], "output": "value_4",
        "elements": 131072, "grid_size": 512,
        "a_shape": [1, 32, 1024], "b_shape": [1024, 4096],
        "c_shape": [1, 32, 4096], "m": 32, "n": 4096, "k": 1024,
    },
    {
        "kernel": "gac_op_1", "kind": "matmul",
        "inputs": ["hidden_states", "value_2"], "output": "value_5",
        "elements": 131072, "grid_size": 512,
        "a_shape": [1, 32, 1024], "b_shape": [1024, 4096],
        "c_shape": [1, 32, 4096], "m": 32, "n": 4096, "k": 1024,
    },
    {
        "kernel": "gac_op_2", "kind": "pointwise",
        "inputs": ["value_4", "value_5"], "output": "value_6",
        "elements": 131072, "grid_size": 1024, "block_elements": 128,
    },
    {
        "kernel": "gac_op_3", "kind": "matmul",
        "inputs": ["value_6", "value_3"], "output": "output",
        "elements": 32768, "grid_size": 128,
        "a_shape": [1, 32, 4096], "b_shape": [4096, 1024],
        "c_shape": [1, 32, 1024], "m": 32, "n": 1024, "k": 4096,
    },
]

EXPECTED_TENSORS = {
    "hidden_states": {"shape": [1, 32, 1024], "dtype": "float32", "bytes": 131072, "storage": "input"},
    "value_1": {"shape": [1024, 4096], "dtype": "float32", "bytes": 16777216, "storage": "constant", "file": "tensor_1.bin"},
    "value_2": {"shape": [1024, 4096], "dtype": "float32", "bytes": 16777216, "storage": "constant", "file": "tensor_2.bin"},
    "value_3": {"shape": [4096, 1024], "dtype": "float32", "bytes": 16777216, "storage": "constant", "file": "tensor_3.bin"},
    "value_4": {"shape": [1, 32, 4096], "dtype": "float32", "bytes": 524288, "storage": "arena"},
    "value_5": {"shape": [1, 32, 4096], "dtype": "float32", "bytes": 524288, "storage": "arena"},
    "value_6": {"shape": [1, 32, 4096], "dtype": "float32", "bytes": 524288, "storage": "arena"},
    "output": {"shape": [1, 32, 1024], "dtype": "float32", "bytes": 131072, "storage": "arena"},
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_file(path: Path, description: str) -> Path:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"missing {description}: {path}")
    return path


def require_exact_hash(path: Path, expected: str, description: str) -> str:
    actual = file_sha256(path)
    if actual != expected:
        raise ValueError(
            f"{description} hash mismatch: expected {expected}, got {actual}: {path}")
    return actual


def validate_plan(plan: dict) -> None:
    root_expected = {
        "schema_version": 1,
        "target": "ascend",
        "execution": "artifact-only",
        "artifact_format": "standard-ttir",
        "runtime_available": False,
        "triton_dispatches": 4,
    }
    for key, expected in root_expected.items():
        if plan.get(key) != expected:
            raise ValueError(f"plan {key!r}: expected {expected!r}, got {plan.get(key)!r}")
    rows = plan.get("dispatches")
    if not isinstance(rows, list) or len(rows) != 4:
        raise ValueError("official plan must contain exactly four dispatches")
    for index, (row, expected) in enumerate(zip(rows, EXPECTED_DISPATCHES)):
        triton_contract = row.get("triton")
        if not isinstance(triton_contract, dict):
            raise ValueError(f"dispatch {index} has no Triton contract")
        direct_fields = ("kernel", "inputs", "output", "elements")
        for key in direct_fields:
            if row.get(key) != expected[key]:
                raise ValueError(
                    f"dispatch {index} {key}: expected {expected[key]!r}, got {row.get(key)!r}")
        if row.get("lowering") != "linalg-to-triton":
            raise ValueError(f"dispatch {index} is not linalg-to-triton")
        if triton_contract.get("artifact_format") != "standard-ttir":
            raise ValueError(f"dispatch {index} is not standard TTIR")
        if triton_contract.get("kind") != expected["kind"]:
            raise ValueError(f"dispatch {index} kind mismatch")
        if triton_contract.get("ttir_file") != expected["kernel"] + ".ttir":
            raise ValueError(f"dispatch {index} TTIR filename mismatch")
        if triton_contract.get("grid_size") != expected["grid_size"]:
            raise ValueError(f"dispatch {index} grid mismatch")
        if expected["kind"] == "matmul":
            for key in ("a_shape", "b_shape", "c_shape", "m", "n", "k"):
                if triton_contract.get(key) != expected[key]:
                    raise ValueError(f"dispatch {index} {key} mismatch")
            if (triton_contract.get("block_m"), triton_contract.get("block_n"),
                    triton_contract.get("block_k")) != (16, 16, expected["k"]):
                raise ValueError(f"dispatch {index} tile mismatch")
            if triton_contract.get("ttir_signature") != \
                    "ptr<A:f32>, ptr<B:f32>, ptr<C:f32>":
                raise ValueError(f"dispatch {index} pointer signature mismatch")
        elif triton_contract.get("block_elements") != expected["block_elements"]:
            raise ValueError(f"dispatch {index} pointwise block mismatch")

    tensors = plan.get("tensors")
    if not isinstance(tensors, list) or {row.get("name") for row in tensors} != set(EXPECTED_TENSORS):
        raise ValueError("plan tensor set does not match the official MLP contract")
    for row in tensors:
        expected = EXPECTED_TENSORS[row["name"]]
        for key, value in expected.items():
            if row.get(key) != value:
                raise ValueError(
                    f"tensor {row['name']} {key}: expected {value!r}, got {row.get(key)!r}")


def normalize_ttir(source: Path, destination: Path) -> None:
    from triton._C.libtriton import buffer_ir, ir
    from triton._C.libtriton.ascend import ir as ascend_ir
    from triton.compiler.compiler import make_backend
    from triton.runtime.driver import driver

    context = ir.context()
    ir.load_dialects(context)
    buffer_ir.load_dialects(context)
    ascend_ir.load_dialects(context)
    backend = make_backend(driver.active.get_current_target())
    backend.load_dialects(context)
    module = ir.parse_mlir_module(str(source), context)
    module.context = context
    destination.write_text(str(module), encoding="utf-8")
    context.disable_multithreading()


def json_metadata(metadata) -> dict:
    result = {}
    for key in metadata._fields:
        value = getattr(metadata, key)
        if key == "target":
            value = {"backend": value.backend, "arch": value.arch,
                     "warp_size": value.warp_size}
        elif isinstance(value, tuple):
            value = list(value)
        result[key] = value
    return result


def load_weight(torch, path: Path, shape: tuple[int, ...]):
    count = math.prod(shape)
    tensor = torch.from_file(
        str(path), shared=False, size=count, dtype=torch.float32).clone().reshape(shape)
    if not tensor.is_contiguous() or not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"constant is non-contiguous or non-finite: {path}")
    return tensor


def compare(torch, actual, reference) -> dict:
    difference = (actual - reference).abs()
    denominator = reference.abs().clamp_min(1e-12)
    return {
        "allclose_rtol1e-5_atol1e-5": bool(
            torch.allclose(actual, reference, rtol=1e-5, atol=1e-5)),
        "allclose_rtol1e-4_atol1e-4": bool(
            torch.allclose(actual, reference, rtol=1e-4, atol=1e-4)),
        "max_abs": float(difference.max()),
        "mean_abs": float(difference.mean()),
        "max_rel_clamped": float((difference / denominator).max()),
        "actual_min": float(actual.min()),
        "actual_max": float(actual.max()),
        "reference_min": float(reference.min()),
        "reference_max": float(reference.max()),
        "finite": bool(torch.isfinite(actual).all()),
    }


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--ttir-dir", type=Path, required=True)
    parser.add_argument("--constants-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True,
                        help="new output directory; existing paths are rejected")
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20261113)
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    plan_path = require_file(arguments.plan, "official plan")
    ttir_dir = arguments.ttir_dir.expanduser().resolve()
    constants_dir = arguments.constants_dir.expanduser().resolve()
    output = arguments.output.expanduser().resolve()
    cache_dir = arguments.cache_dir.expanduser().resolve()
    if output.exists():
        raise ValueError(f"output already exists: {output}")
    output.mkdir(parents=True)
    normalized_dir = output / "normalized"
    compiled_dir = output / "compiled"
    normalized_dir.mkdir()
    compiled_dir.mkdir()

    os.environ["TRITON_CACHE_DIR"] = str(cache_dir)
    os.environ["TRITON_ALWAYS_COMPILE"] = "1"

    # Load torch_npu before initializing the active Triton driver.
    import torch
    import torch_npu
    import triton
    from triton.compiler import compile as triton_compile
    from triton.runtime.driver import driver

    plan_hash = require_exact_hash(
        plan_path, OFFICIAL_PLAN_SHA256, "official plan")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    validate_plan(plan)

    target = driver.active.get_current_target()
    if target.backend != "npu" or not str(target.arch).startswith("Ascend910B"):
        raise ValueError(f"expected an Ascend 910B NPU target, got {target}")
    torch.npu.set_device(arguments.device)
    if "Ascend910B" not in torch.npu.get_device_name(arguments.device):
        raise ValueError(
            f"expected an Ascend 910B device, got {torch.npu.get_device_name(arguments.device)!r}")

    source_hashes = {}
    constant_hashes = {}
    for kernel, expected_hash in OFFICIAL_TTIR_SHA256.items():
        source = require_file(ttir_dir / f"{kernel}.ttir", f"{kernel} TTIR")
        source_hashes[kernel] = require_exact_hash(
            source, expected_hash, f"{kernel} TTIR")
    for name, expected_hash in OFFICIAL_CONSTANT_SHA256.items():
        path = require_file(constants_dir / name, name)
        if path.stat().st_size != 16 * 1024 * 1024:
            raise ValueError(f"unexpected constant size: {path}")
        constant_hashes[name] = require_exact_hash(path, expected_hash, name)

    kernels = []
    compile_reports = []
    for dispatch in EXPECTED_DISPATCHES:
        name = dispatch["kernel"]
        source = ttir_dir / f"{name}.ttir"
        normalized = normalized_dir / f"{name}.ttir"
        normalize_ttir(source, normalized)
        kernel = triton_compile(str(normalized))
        signature_before = dict(kernel.src.signature)
        expected_bad_signature = {0: "*f32>", 1: "*f32>", 2: "*f32>"}
        if signature_before != expected_bad_signature:
            raise ValueError(
                f"unexpected Triton 3.2 IRSource signature for {name}: {signature_before}")
        kernel.src.signature = {0: "*fp32", 1: "*fp32", 2: "*fp32"}
        if kernel.metadata.kernel_name != name:
            raise ValueError(
                f"compiled kernel name changed: {name} -> {kernel.metadata.kernel_name}")
        if set(kernel.asm) != {"ttadapter", "npubin"}:
            raise ValueError(f"unexpected assembly stages for {name}: {sorted(kernel.asm)}")
        npubin = bytes(kernel.asm["npubin"])
        expected_size, expected_hash = OFFICIAL_NPUBIN[name]
        actual_hash = hashlib.sha256(npubin).hexdigest()
        if len(npubin) != expected_size or actual_hash != expected_hash:
            raise ValueError(
                f"{name} npubin mismatch: expected {expected_size}/{expected_hash}, "
                f"got {len(npubin)}/{actual_hash}")
        (compiled_dir / f"{name}.npubin").write_bytes(npubin)
        (compiled_dir / f"{name}.ttadapter.mlir").write_text(
            kernel.asm["ttadapter"], encoding="utf-8")
        metadata = json_metadata(kernel.metadata)
        (compiled_dir / f"{name}.metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        compile_reports.append({
            "kernel": name,
            "grid": [dispatch["grid_size"], 1, 1],
            "source_sha256": source_hashes[name],
            "normalized_sha256": file_sha256(normalized),
            "signature_before": signature_before,
            "signature_after": dict(kernel.src.signature),
            "npubin_bytes": len(npubin),
            "npubin_sha256": actual_hash,
        })
        kernels.append(kernel)

    w1 = load_weight(torch, constants_dir / "tensor_1.bin", (1024, 4096))
    w2 = load_weight(torch, constants_dir / "tensor_2.bin", (1024, 4096))
    w3 = load_weight(torch, constants_dir / "tensor_3.bin", (4096, 1024))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(arguments.seed)
    x = torch.randn((1, 32, 1024), generator=generator, dtype=torch.float32)
    x2 = x.reshape(32, 1024)

    cpu_started = time.perf_counter()
    gate_reference = x2 @ w1
    up_reference = x2 @ w2
    exponent = torch.exp(-torch.abs(gate_reference))
    sigmoid_reference = torch.where(
        gate_reference < 0,
        exponent / (1 + exponent),
        1 / (1 + exponent),
    )
    fused_reference = gate_reference * sigmoid_reference * up_reference
    output_reference = (fused_reference @ w3).reshape(1, 32, 1024)
    cpu_seconds = time.perf_counter() - cpu_started

    torch.npu.reset_peak_memory_stats(arguments.device)
    x_npu = x.to("npu")
    w1_npu = w1.to("npu")
    w2_npu = w2.to("npu")
    w3_npu = w3.to("npu")
    gate_npu = torch.empty((1, 32, 4096), dtype=torch.float32, device="npu")
    up_npu = torch.empty_like(gate_npu)
    fused_npu = torch.empty_like(gate_npu)
    output_npu = torch.empty((1, 32, 1024), dtype=torch.float32, device="npu")
    torch.npu.synchronize()
    npu_started = time.perf_counter()
    kernels[0][(512, 1, 1)](x_npu, w1_npu, gate_npu)
    kernels[1][(512, 1, 1)](x_npu, w2_npu, up_npu)
    kernels[2][(1024, 1, 1)](gate_npu, up_npu, fused_npu)
    kernels[3][(128, 1, 1)](fused_npu, w3_npu, output_npu)
    torch.npu.synchronize()
    npu_seconds = time.perf_counter() - npu_started

    gate = gate_npu.cpu()
    up = up_npu.cpu()
    fused = fused_npu.cpu()
    output_tensor = output_npu.cpu()
    checks = {
        "gac_op_0_gate": compare(
            torch, gate, gate_reference.reshape_as(gate)),
        "gac_op_1_up": compare(
            torch, up, up_reference.reshape_as(up)),
        "gac_op_2_fused": compare(
            torch, fused, fused_reference.reshape_as(fused)),
        "gac_op_3_output": compare(torch, output_tensor, output_reference),
    }
    passed = all(row["finite"] and row["allclose_rtol1e-5_atol1e-5"]
                 for row in checks.values())
    report = {
        "kind": "official-gac-mlp-validation-launcher",
        "scope": "pinned validation only; not a general GAC runtime",
        "status": "passed" if passed else "failed",
        "plan": str(plan_path),
        "plan_sha256": plan_hash,
        "ttir_dir": str(ttir_dir),
        "constants_dir": str(constants_dir),
        "cache_dir": str(cache_dir),
        "strict_official_hashes_validated": True,
        "target": {"backend": target.backend, "arch": target.arch,
                   "warp_size": target.warp_size},
        "device": torch.npu.get_device_name(arguments.device),
        "device_index": arguments.device,
        "versions": {
            "torch": torch.__version__,
            "torch_npu": torch_npu.__version__,
            "triton_imported": triton.__version__,
            "triton_distribution": importlib.metadata.version("triton"),
            "triton_ascend_distribution": importlib.metadata.version("triton_ascend"),
        },
        "input": {"shape": list(x.shape), "dtype": "float32",
                  "seed": arguments.seed, "min": float(x.min()),
                  "max": float(x.max())},
        "source_hashes": source_hashes,
        "constant_hashes": constant_hashes,
        "dispatch_order": [row["kernel"] for row in EXPECTED_DISPATCHES],
        "compiled": compile_reports,
        "checks": checks,
        "cpu_oracle_seconds": cpu_seconds,
        "npu_four_dispatch_seconds_including_first_launch": npu_seconds,
        "timing_note": "first launch includes lazy launcher/module initialization; not a benchmark",
        "peak_memory_allocated_bytes": int(
            torch.npu.max_memory_allocated(arguments.device)),
        "peak_memory_reserved_bytes": int(
            torch.npu.max_memory_reserved(arguments.device)),
    }
    result_path = output / "validation.json"
    result_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True), flush=True)
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())

